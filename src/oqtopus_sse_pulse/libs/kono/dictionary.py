def check_keys(input_dict: dict, required_keys: list):
    """
    指定された辞書に必要なキーが存在するか確認し、存在しない場合は警告を表示します。

    Parameters:
    input_dict (dict): キーの存在を確認する辞書。
    required_keys (list): 確認する必要があるキーのリスト。
    
    Raises:
    KeyError: 必要なキーが存在しない場合に発生。
    """
    for key in required_keys:
        if key not in input_dict:
            print(f"Warning: 辞書に'{key}'キーがありません。")
            raise KeyError(f"Missing required key: {key}")