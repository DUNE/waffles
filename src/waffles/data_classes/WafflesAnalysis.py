from abc import ABC, abstractmethod
import waffles.Exceptions as we
import argparse

class WafflesAnalysis(ABC):

    read_input_itr: int
    analyze_itr: int

    def __init__(self):


        # self.path_to_input_file 
        # self.path_to_output_file 
        
        # self.read_input_loop = self.define_read_input_loop()
        # self.analyze_loop

        # self.analyze_itr    = None 
        # self.read_input_itr = None
        


        pass

    
    # @abstractmethod
    # def define_read_input_loop():
    #     pass
    
    @abstractmethod
    def arguments(parse: argparse.ArgumentParser):
        pass

    @abstractmethod
    def initialize(args: list):
        pass

    @abstractmethod
    def read_input()->bool:
        pass

    @abstractmethod
    def analyze()->bool:
        pass

    @abstractmethod    
    def write_output():
        pass

###########################################
    def base_arguments(self, first_arg: str):

        parse = argparse.ArgumentParser()

        if first_arg != '-i' and first_arg != '-h' :
            parse.add_argument('analysis',           type=str,                 help="name of the analysis to process")

        # input and output files
        parse.add_argument('-i','--input_file',  type=str, required=False, help="path of input file or folder")
        parse.add_argument('-o','--output_file', type=str, required=False, help="path to output file", default = None)

        # call the method of the derived class to add additional arguments
        self.arguments(parse)

        # parse the arguments 
        args = vars(parse.parse_args())

        return args

###########################################
    def base_initialize(self, args):        

        # asign the input and output files to data members
        self.path_to_input_file = args['input_file']
        self.path_to_output_file = args['output_file']

        print ('Input  file:', self.path_to_input_file)
        print ('Output file:', self.path_to_output_file)

        # by default there is no iteration (only one elemet in list)
        self.read_input_loop =[0]
        self.analyze_loop =[0]

        # call the method of the derived class
        self.initialize(args)

###########################################        
    def execute(self,first_arg: str):

        args = self.base_arguments(first_arg)
        self.base_initialize(args)

        for self.read_input_itr in self.read_input_loop:
            print (f'read_input loop: {self.read_input_itr}' )
            if not self.read_input():
                continue

            for self.analyze_itr in self.analyze_loop:
                print (f'analyze loop: {self.analyze_itr}' )
                if not self.analyze():
                    continue

                if self.path_to_output_file != None: 
                    self.write_output()
    