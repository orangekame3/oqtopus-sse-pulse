REQUIRED_KEYS = {
    "qubit_settings": ["chip_id", "muxes", "qubit"],
    "config_file_info": ["params_dir", "calib_note_path"]
}

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
        

def check_qubit_settings(qubit_settings: dict):
    """
    qubit_settings辞書に必要なキーが存在するか確認します。

    Parameters:
    qubit_settings (dict): qubit_settings辞書。
    
    Raises:
    KeyError: 必要なキーが存在しない場合に発生。
    """
    required_keys = REQUIRED_KEYS["qubit_settings"]
    check_keys(input_dict=qubit_settings, required_keys=required_keys)


def check_config_file_info(config_file_info: dict):
    """
    config_file_info辞書に必要なキーが存在するか確認します。

    Parameters:
    config_file_info (dict): config_file_info辞書。
    
    Raises:
    KeyError: 必要なキーが存在しない場合に発生。
    """
    required_keys = REQUIRED_KEYS["config_file_info"]
    check_keys(input_dict=config_file_info, required_keys=required_keys)