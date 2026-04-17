import re

from app.agents.conversational.state import CallState
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def norm_text_node(state: CallState) -> dict:
    text = state["raw_transcript"].strip()
    text = re.sub(r"\s+", " ", text)
    return {"normalized_text": text}
