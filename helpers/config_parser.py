def read_config(config_path: str):
    content = ''
    
    with open(config_path, 'r') as file:
        content = file.readlines()
    
    return {config.split('=')[0]: int(config.split('=')[1]) for config in content}