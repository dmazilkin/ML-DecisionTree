def read_config(config_path: str):
    content = ''
    
    with open(config_path, 'r') as file:
        content = file.readlines()
    
    configs = dict()
    
    for config in content:
        name = config.split('=')[0]
        value = config.split('=')[1]
        try:
            value = int(value)
        except:
            if name.lower() == 'bins':
                value = None
        finally:
            configs[name] = value
            
    return configs