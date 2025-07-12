from argparse import ArgumentParser

class CLIParser:
    def __init__(self) -> None:
        self.parser = ArgumentParser()
        self.parser.add_argument('-e', '--example', required=True)
        self.parser.add_argument('-c', '--config', required=True)
    
    def parse_cli(self) -> dict:
        args = self.parser.parse_args()
        
        return vars(args)