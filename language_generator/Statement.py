import json
class Statement:
    def __init__(self, sentence, relationship_type, relationship):

        self.sentence = sentence

        self.relationship_type = relationship_type
        self.relationship = relationship

        self.target_object_name = None
        self.target_object_name = None
        self.target_object_name = None

        self.target_color_used = False
        self.target_size_used = False
        self.anchor_color_used = False
        self.anchor_size_used = False
        self.anchor1_color_used = False
        self.anchor1_size_used = False
        self.anchor2_color_used = False
        self.anchor2_size_used = False

    def replace(self, a, b):
        self.sentence = self.sentence.replace(a, b)

    def to_json(self):
        return json.dumps(self.__dict__)