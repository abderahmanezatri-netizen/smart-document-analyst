from sda.tools.schemas import ClassificationOutput

def test_classification_schema():
    obj = ClassificationOutput(predicted_class='invoice', confidence=0.9, class_probabilities={'invoice':0.9}, low_confidence=False, human_review_required=False)
    assert obj.predicted_class == 'invoice'
