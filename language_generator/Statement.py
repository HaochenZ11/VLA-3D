import json
class Statement:
    def __init__(self, sentence, relationship_type, relationship):

        self.sentence = sentence

        self.relationship_type = relationship_type
        self.relationship = relationship

        self.target_object_name = None
        self.target_object_name = None
        self.target_object_name = None

        if self.relationship_type == "binary" or self.relationship_type == "ordered":
            self.anchor_object_name = None
            self.anchor_object_name = None
            self.anchor_object_name = None
        elif self.relationship_type == "ternary":
            self.anchor1_object_name = None
            self.anchor1_object_name = None
            self.anchor1_object_name = None
            self.anchor2_object_name = None
            self.anchor2_object_name = None
            self.anchor2_object_name = None

    def replace(self, a, b):
        self.sentence = self.sentence.replace(a, b)

    def to_json(self):
        return json.dumps(self.__dict__)