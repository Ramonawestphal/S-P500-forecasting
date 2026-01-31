def extract_params(result_dict):
    assert isinstance(result_dict, dict)
    assert "params" in result_dict
    return result_dict["params"]

