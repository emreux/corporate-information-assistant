"""
CIA web interface (Streamlit): sign in, chat with the documents, see the sources, manage the documents.

Only presentation lives here. Answering, indexing, the catalog and the users are in rag/, and every
text shown on screen is in ui_texts.toml. What a user may do is checked inside the functions that do
it (rag.auth.require_admin), not only by hiding buttons.

Run from the project root:
    streamlit run app.py
"""
import logging
import time
import tomllib
from pathlib import Path

import streamlit as st

import config
from rag.answer import Answer, ConfidentialModelUnavailable, answer_question, build_messages
from rag.auth import AuthError, User, authenticate, load_users, require_admin
from rag.citations import best_excerpt
from rag.llm import LLMError, LLMUnavailable
from rag.model_client import ModelServerError, ModelServerUnavailable, get_model_client
from rag.query_log import log_question
from rag.retriever import Hit, RetrievalError
from rag.store import get_qdrant

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


# Read on every rerun: the file is tiny, and an edited text shows up on the next click.
# (Heavy objects such as the Qdrant client already live in rag/ as lru_cache singletons. Streamlit
# reruns this script, not the imported modules, so they are created only once per server process.)
with open(config.UI_TEXTS, "rb") as texts_file:
    TEXTS: dict[str, str] = tomllib.load(texts_file)


# --- Helpers ------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def system_status() -> dict:
    """Check Qdrant and the model server at most every 30 seconds, not on every click."""
    status = {"points": None, "model_server": False}
    try:
        status["points"] = get_qdrant().count(config.QDRANT_COLLECTION, exact=True).count
    except Exception:
        pass
    try:
        get_model_client().health()
        status["model_server"] = True
    except Exception:
        pass
    return status


def error_text(exc: Exception) -> str:
    """A message the user can act on, instead of a Python traceback."""
    if isinstance(exc, ModelServerUnavailable):
        return TEXTS["error_model_server"]
    if isinstance(exc, ModelServerError):
        return TEXTS["error_model_server_other"].format(detail=exc)
    if isinstance(exc, RetrievalError):
        return TEXTS["error_retrieval"]
    if isinstance(exc, ConfidentialModelUnavailable):
        return TEXTS["error_confidential_model"]
    if isinstance(exc, LLMUnavailable):
        return TEXTS["error_llm_unavailable"]
    if isinstance(exc, LLMError):
        return TEXTS["error_llm"].format(detail=exc)
    return TEXTS["error_unexpected"].format(detail=f"{type(exc).__name__}: {exc}")


def source_line(number: int, hit: Hit) -> str:
    """One source in the list under an answer, in the language of the interface."""
    where = " › ".join(part for part in (hit.source, hit.heading_path) if part)
    details = []
    if hit.page_start is not None:
        details.append(TEXTS["page_single"].format(page=hit.page_start) if hit.page_start == hit.page_end
                       else TEXTS["page_range"].format(start=hit.page_start, end=hit.page_end))
    details.append(TEXTS["score"].format(score=hit.score))
    if hit.status == "superseded":
        details.append(TEXTS["superseded_marker"])
    if hit.ocr:
        details.append(TEXTS["ocr_marker"])
    return f"**[{number}]** {where} · " + " · ".join(details)


def unverified_text(answer: Answer) -> str:
    """The numbers and codes of the answer that its cited sources do not contain, for a warning."""
    items = []
    for item in answer.unverified:
        if item.found_in:
            items.append(TEXTS["unverified_elsewhere"].format(
                fact=item.fact, sources=", ".join(f"[{number}]" for number in item.found_in)))
        else:
            items.append(f"**{item.fact}**")
    return ", ".join(items)


def show_answer_details(answer: Answer) -> None:
    """The sources the answer cites, warnings about its citations, and the developer views."""
    if answer.rewritten:                               # how a follow-up question was understood (step 17)
        st.caption(TEXTS["rewritten_note"].format(question=answer.search_question))
    citations = answer.citations
    cited = answer.cited_sources()
    if citations.not_found and answer.llm:
        st.caption(TEXTS["not_found_note"])
    elif not cited and answer.llm:
        st.warning(TEXTS["uncited_warning"])            # facts without a source: not checkable
    if citations.invalid:
        st.warning(TEXTS["invalid_citation_warning"].format(
            numbers=", ".join(f"[{number}]" for number in citations.invalid)))
    if answer.unverified:                              # step 19: numbers that the cited sources do not contain
        st.warning(TEXTS["unverified_warning"].format(facts=unverified_text(answer)))
    if cited:
        with st.expander(TEXTS["sources_title"].format(count=len(cited))):
            for number, hit in cited:
                st.markdown(source_line(number, hit))
                st.caption(best_excerpt(hit.content, citations.clean_text, config.SOURCE_EXCERPT_CHARS))

    if answer.confidential:
        st.caption(TEXTS["confidential_note"])

    if st.session_state.get("developer_mode") and answer.sources:
        unused = answer.unused_sources()
        if unused:                                     # what the search brought in but the answer did not need
            with st.expander(TEXTS["unused_title"].format(count=len(unused))):
                for number, hit in unused:
                    st.markdown(source_line(number, hit))
        with st.expander(TEXTS["context_title"]):
            for message in build_messages(answer.search_question, answer.sources):
                st.code(f"[{message['role']}]\n{message['content']}", language=None, wrap_lines=True)
    if answer.llm:
        st.caption(TEXTS["answer_meta"].format(model=answer.llm.model, seconds=answer.llm.elapsed_ms / 1000,
                                               retrieval_ms=answer.retrieval_ms))
        if answer.llm.used_fallback:
            st.warning(TEXTS["fallback_warning"].format(reason=answer.llm.fallback_reason))


def show_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["error"])
            return
        st.markdown(message["content"])
        if message.get("answer"):
            show_answer_details(message["answer"])


# --- Document management ------------------------------------------------------------
# The indexer is imported inside these functions: it loads the document reader (Docling),
# which takes several seconds, and chatting does not need it.

def handle_upload(uploaded_file, access_level: str) -> None:
    from rag.indexer import IGNORED_PREFIXES, IndexerError, index_file
    from rag.metadata import CatalogError, add_catalog_entry, load_catalog
    from rag.reader import UnsupportedFormatError, check_supported

    try:
        require_admin(current_user())
    except PermissionError:
        st.error(TEXTS["error_not_allowed"])
        return
    target = config.SAMPLES_DIR / Path(uploaded_file.name).name     # never trust a path sent by the browser
    if target.name.startswith(IGNORED_PREFIXES):                    # the folder sync would delete it again
        st.error(TEXTS["upload_ignored_name"].format(name=target.name))
        return
    try:
        check_supported(target)
    except UnsupportedFormatError as exc:
        st.error(TEXTS["upload_unsupported"].format(detail=exc))
        return

    replaced = target.exists()
    with st.spinner(TEXTS["indexing"].format(name=target.name)):
        try:
            target.write_bytes(uploaded_file.getvalue())
            catalog = load_catalog()
            if target.name not in catalog:
                add_catalog_entry(target.name, access_level)
                catalog = load_catalog()
            result = index_file(target, catalog, get_qdrant())
        except CatalogError as exc:
            st.error(TEXTS["error_catalog"].format(detail=exc))
            return
        except IndexerError as exc:
            st.error(TEXTS["error_indexer"].format(detail=exc))
            return
        except Exception as exc:
            logger.exception("Upload of %s failed", target.name)
            st.error(error_text(exc))
            return

    st.success(TEXTS["upload_done"].format(name=target.name, chunks=result.chunks))
    if replaced:
        st.info(TEXTS["upload_replaced"])
    stored_level = catalog[target.name].access_level          # the catalog decides, not the radio button
    if stored_level != access_level:
        st.info(TEXTS["upload_existing_label"].format(level=TEXTS[f"access_{stored_level}"]))
    if result.detail:
        st.warning(TEXTS["upload_warnings"].format(detail=result.detail))
    system_status.clear()


def handle_sync() -> None:
    from rag.indexer import IndexerError, sync_folder
    from rag.metadata import CatalogError, load_catalog

    try:
        require_admin(current_user())
    except PermissionError:
        st.error(TEXTS["error_not_allowed"])
        return
    with st.spinner(TEXTS["syncing"]):
        try:
            results = sync_folder(config.SAMPLES_DIR, load_catalog(), get_qdrant())
        except CatalogError as exc:
            st.error(TEXTS["error_catalog"].format(detail=exc))
            return
        except IndexerError as exc:
            st.error(TEXTS["error_indexer"].format(detail=exc))
            return
        except Exception as exc:
            logger.exception("Folder sync failed")
            st.error(error_text(exc))
            return

    st.dataframe([{TEXTS["col_file"]: result.source,
                   TEXTS["col_result"]: TEXTS.get(f"action_{result.action}", result.action),
                   TEXTS["col_chunks"]: str(result.chunks) if result.chunks else "",   # one type per column
                   TEXTS["col_detail"]: result.detail} for result in results], hide_index=True)
    system_status.clear()


# --- Signing in ---------------------------------------------------------------------

def current_user() -> User | None:
    """
    The signed-in user, read again from the user file on every click: a removed user or a changed
    role takes effect at once. A broken user file signs everyone out (fail closed).
    """
    signed_in = st.session_state.get("user")
    if signed_in is None:
        return None
    try:
        entry = load_users().get(signed_in.username)
    except AuthError:
        return None
    return entry[0] if entry else None


def show_login() -> None:
    """The login form. Nothing after it runs until someone has signed in."""
    st.title(TEXTS["page_title"])
    try:
        has_users = bool(load_users())
    except AuthError as exc:
        st.error(TEXTS["error_users_file"].format(detail=exc))
        st.stop()
    if not has_users:
        st.warning(TEXTS["login_no_users"])
        st.stop()

    with st.form("login"):
        st.subheader(TEXTS["login_title"])
        username = st.text_input(TEXTS["login_username"])
        password = st.text_input(TEXTS["login_password"], type="password")
        submitted = st.form_submit_button(TEXTS["login_button"])
    if submitted:
        user = authenticate(username, password)
        if user is None:
            time.sleep(config.LOGIN_FAILURE_DELAY)
            st.error(TEXTS["login_failed"])           # never say whether the name or the password was wrong
        else:
            st.session_state.clear()
            st.session_state.user = user
            st.rerun()
    st.caption(TEXTS["login_hint"])
    st.stop()


# --- Page ---------------------------------------------------------------------------

st.set_page_config(page_title=TEXTS["page_title"], page_icon="🛡️", layout="wide")
user = current_user()
if user is None:
    if "user" in st.session_state:       # signed in before, but removed since (or the user file is broken)
        st.session_state.clear()         # (never clear before a login: it would drop the submitted form)
    show_login()
if "messages" not in st.session_state:
    st.session_state.messages = []       # survives the rerun that Streamlit does after every click

with st.sidebar:
    st.markdown(TEXTS["signed_in_as"].format(name=user.name, role=TEXTS[f"role_{user.role}"]))
    if st.button(TEXTS["logout_button"]):
        st.session_state.clear()         # the next person at this browser must not see this conversation
        st.rerun()

    st.header(TEXTS["status_title"])
    status = system_status()
    if status["points"] is None:
        st.markdown(TEXTS["status_qdrant_down"])
    else:                                # the number of chunks counts confidential ones too: admins only
        st.markdown(TEXTS["status_qdrant_ok"].format(points=status["points"]) if user.is_admin
                    else TEXTS["status_qdrant_ready"])
    st.markdown(TEXTS["status_model_ok"] if status["model_server"] else TEXTS["status_model_down"])

    st.header(TEXTS["settings_title"])
    st.checkbox(TEXTS["include_superseded"], key="include_superseded")
    if user.is_admin:
        st.checkbox(TEXTS["developer_mode"], key="developer_mode")
    if st.button(TEXTS["clear_chat"]):
        st.session_state.messages = []
        st.rerun()

    if user.is_admin:
        st.header(TEXTS["documents_title"])
        uploaded = st.file_uploader(TEXTS["upload_label"])
        level = st.radio(TEXTS["access_label"], options=config.ACCESS_LEVELS,     # starts at "restricted":
                         index=config.ACCESS_LEVELS.index(config.DEFAULT_ACCESS_LEVEL),   # sharing is a choice
                         format_func=lambda value: TEXTS[f"access_{value}"])
        if st.button(TEXTS["upload_button"], disabled=uploaded is None):
            handle_upload(uploaded, level)
        if st.button(TEXTS["sync_button"]):
            handle_sync()

st.title(TEXTS["page_title"])
st.caption(TEXTS["app_intro"])

for message in st.session_state.messages:
    show_message(message)

question = st.chat_input(TEXTS["chat_placeholder"])
if question:
    show_message({"role": "user", "content": question})
    # Step 17: the earlier questions, as they were searched (a rewritten follow-up is clearer than "Peki ya?")
    history = [message.get("searched_as", message["content"])
               for message in st.session_state.messages if message["role"] == "user"]
    asked = {"role": "user", "content": question, "searched_as": question}
    with st.chat_message("assistant"):
        with st.spinner(TEXTS["thinking"]):
            try:
                answer = answer_question(question, include_superseded=st.session_state.include_superseded,
                                         access_levels=user.access_levels, history=history)
                reply = {"role": "assistant", "answer": answer,       # clean_text: without [NOT_FOUND]
                         "content": answer.citations.clean_text if answer.llm else TEXTS["no_sources"]}
                asked["searched_as"] = answer.search_question
                log_question(user.username, user.role, question, answer=answer)
            except Exception as exc:
                logger.exception("Answering failed")
                reply = {"role": "assistant", "error": error_text(exc)}
                log_question(user.username, user.role, question, error=exc)
    st.session_state.messages += [asked, reply]
    st.rerun()                           # draw the whole conversation again, now with the new answer
