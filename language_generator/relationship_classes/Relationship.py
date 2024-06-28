from abc import ABC, abstractmethod
class Relationship(ABC):
    def __init__(self, region_graph, objects, objects_class_region, language_master_template, object_filter):
        """
        Parses arguments from command line for input scene graph and language template
            Params:
                region_graph(Dict): Scene data from a region
                language_master_template(Dict): Template of language syntax and sentence structures
                object_filter(ObjectFilter): ObjectFilter object with the scene object data
        """
        self.region_graph = region_graph
        self.language_master_template = language_master_template
        self.object_filter = object_filter
        self.objects = objects
        self.objects_class_region = objects_class_region

    @abstractmethod
    def generate_statements(self, generation_configs, max_statements=None):
        """
        All children od the Relationship class must generate statements
            Params:
                max_statements(int): Maximum amount of statements per relation
            Returns:
                statements(Dict): Dictionary of relational statement and corresponding ground truth data
        """
        pass