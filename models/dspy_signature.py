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

class CFIRClassify(dspy.Signature):
    """
    Classify qualitative comments into CFIR Constructs using a context-based approach.
    """

    # Input Fields
    comments: str = dspy.InputField(
        description="The qualitative comment to classify."
    )
    cfir_context: str = dspy.InputField(
        description="Static context containing descriptions of CFIR constructs."
    )

    # Output Fields
    cfir_construct: list[Literal[tuple(constructs)]] = dspy.OutputField(
        description="The CFIR constructs predicted for the input comment."
    )
    confidence: float = dspy.OutputField(
        description="The confidence score of the prediction (0.0 to 1.0)."
    )

classify = dspy.Predict(Categorize)