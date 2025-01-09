import dspy 
from typing import Literal 
from utils.data_preprocessing import *
from utils.config import *

descriptions, constructs = extract_construct_descriptions(DOMAIN, CFIR_CONSTRUCTS_FILE)

class Categorize(dspy.Signature):
    """Classify comments into CFIR Construct"""

    comments: str = dspy.InputField()
    cfir_context: str = dspy.InputField()  # NEW: Provide the CFIR descriptions/construct context
    cfir_construct: list[Literal[tuple(constructs)]] = dspy.OutputField()
    confidence: float = dspy.OutputField()

classify = dspy.Predict(Categorize)