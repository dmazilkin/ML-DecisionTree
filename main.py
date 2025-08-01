from examples.classification.classification_example import classification
from examples.stepik.stepik_classification import stepik_classification
from helpers.config_parser import read_config
from helpers.cli_parser import CLIParser

def main():
    cli_parser = CLIParser()
    args = cli_parser.parse_cli()
    example, config = args['example'], args['config']
    example_config = read_config(config)
    
    if example == 'classification':
        classification(example_config)
    elif example == 'stepik':
        stepik_classification(example_config)
    else:
        raise('Unknown example.')

if __name__ == '__main__':
    main()