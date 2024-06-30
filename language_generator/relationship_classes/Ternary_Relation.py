from object_filtering import ObjectFilter
from relationship_classes.Relationship import Relationship
from Statement import Statement
import inflect
from multiprocessing import Pool
import time



import json

class Ternary(Relationship):
    '''
    Class for in Between object relationships in scene
    '''
    def __init__(self, region_graph, relation_language_template, objects, objects_class_region, language_master_template, object_filter):
        """
        Parses arguments from command line for input scene graph and language template
           Params:
               region_graph(Dict): Scene data from a region
               relation_langauge_template(Dict): Template of language syntax and relational sentence structures
               language_master_template(Dict): Template of language syntax and object level grammar structures
               object_filter(ObjectFilter): ObjectFilter object with the scene object data
        """

        super().__init__(region_graph, objects, objects_class_region, language_master_template, object_filter)


        self.relation_language_template = relation_language_template
        self.relation = self.relation_language_template['relation']
        if self.relation in self.region_graph:
            self.target_anchors = self.region_graph[self.relation]
        else:
            self.target_anchors = {}

        self.relation_type = 'ternary'


    def generate_statements(self, generation_configs, max_statements=None):
        """
        Generates dataset of in between relational statements with ground truth attributes of the objects
            Params:
                max_statements(int): Maximum amount of statements per relation
            Returns:
                statements(Dict): Dictionary of relational statement and corresponding ground truth data
        """
        statements = {}
        template_sets = self.relation_language_template['templates']


        for t_set in template_sets:
            conditions = t_set["conditions"]

            for target_class, target_ids in self.target_anchors.items():
                if target_class in generation_configs['exclude_classes'] or (target_class == 'space' and not generation_configs['include_spaces']):
                    continue

                for target_id, anchors_ids in target_ids.items():

                    for anchors_set in anchors_ids:
                        anchor1_class = self.objects[anchors_set[0]]['nyu_label']
                        anchor2_class = self.objects[anchors_set[1]]['nyu_label']

                        if anchor1_class == 'space' or anchor2_class == 'space' or anchor1_class in generation_configs[
                            'exclude_classes'] or anchor2_class in generation_configs['exclude_classes']:
                            continue

                        distractors = self.object_filter.get_distractors(target_id, target_class)

                        if self.condition_check(target_id, anchors_set, conditions):
                            statement_candidates = self.get_statement_candidates(t_set, target_id, anchors_set, target_class)
                            for statement in statement_candidates:

                                if statement.sentence not in statements:
                                    statements[statement.sentence] = []
                                target_pos = self.objects[target_id]["center"]
                                target_color = self.objects[target_id]["color_labels"]
                                target_size = self.objects[target_id]["volume"]

                                anchor1_pos = self.objects[anchors_set[0]]["center"]
                                anchor1_color = self.objects[anchors_set[0]]["color_labels"]
                                anchor1_size = self.objects[anchors_set[0]]["volume"]

                                anchor2_pos = self.objects[anchors_set[1]]["center"]
                                anchor2_color = self.objects[anchors_set[1]]["color_labels"]
                                anchor2_size = self.objects[anchors_set[1]]["volume"]


                                statements[statement.sentence].append({
                                    "target_index": target_id,
                                    "target_class": target_class,
                                    "target_position": target_pos,
                                    "target_colors": target_color,
                                    "target_size": target_size,
                                    "target_color_used": statement.target_color_used,
                                    "target_size_used": statement.target_size_used,
                                    "distractor_ids": distractors, "relation": self.relation,
                                    "relation_type": self.relation_type,
                                    "anchors": {
                                        "anchor_1": {
                                            "index": anchors_set[0],
                                            "class": anchor1_class,
                                            "position": anchor1_pos,
                                            "color": anchor1_color,
                                            "size": anchor1_size,
                                            "color_used": statement.anchor1_color_used,
                                            "size_used": statement.anchor1_size_used
                                        },
                                        "anchor_2": {
                                            "index": anchors_set[1],
                                            "class": anchor2_class,
                                            "position": anchor2_pos,
                                            "color": anchor2_color,
                                            "size": anchor2_size,
                                            "color_used": statement.anchor2_color_used,
                                            "size_used": statement.anchor2_size_used
                                        }
                                    }
                                })


            if max_statements is not None and len(statements) == max_statements:
                return statements

        return statements



    def get_statement_candidates(self, template_set, target, anchors, target_class):
        """
        Finds uniquely identifiable objects and combinatorially generates relational statements
            Params:
                template_set(Dict): Sentence template for the relational statements
                target(int): Index of target object
                anchor(int):Index of anchor object
            Returns:
                statement_candidates(List): List of all  relational statements for uniquely identifiable objects
        """
        assert len(template_set["sentences"]) > 0
        statement_candidates = []

        relation_synonyms = self.relation_language_template["synonyms"]["relation"]

        anchor1 = anchors[0]
        anchor2 = anchors[1]


        for sentence in template_set["sentences"]:
            if target_class in self.language_master_template['verb_index']:
                target_verb_index = self.language_master_template['verb_index'][target_class]
            else:
                target_verb_index = 0

            target_verb = self.language_master_template['verbs'][target_verb_index]

            sentence = sentence.replace('%target%', target_class)
            sentence = sentence.replace('%target_verb%', target_verb)
            sentence = sentence.replace('%anchor1%', self.objects[anchor1]['nyu_label'])
            sentence = sentence.replace('%anchor2%', self.objects[anchor2]['nyu_label'])

            p = inflect.engine()
            aux_sentence = template_set["aux_sentences"][0]
            aux_sentence = aux_sentence.replace('%target%', target_class)
            aux_sentence = aux_sentence.replace('%target_verb%', target_verb)
            aux_sentence = aux_sentence.replace('%anchor1%', p.plural(self.objects[anchor1]['nyu_label']))

            object_filter = self.object_filter

            target_colors, target_sizes, anchor1_colors, anchor1_sizes, anchor2_colors, anchor2_sizes = object_filter.filter_targets_and_anchors_ternary(target, anchor1, anchor2, self.relation)

            for target_color in target_colors:
                for anchor1_color in anchor1_colors:
                    for anchor2_color in anchor2_colors:
                        for target_size in target_sizes:
                            for anchor1_size in anchor1_sizes:
                                for anchor2_size in anchor2_sizes:
                                    for relation in relation_synonyms:

                                        statement = Statement(sentence, self.relation_type, self.relation)

                                        statement.target_color_used = target_color
                                        statement.target_size_used = target_size
                                        statement.anchor1_color_used = anchor1_color
                                        statement.anchor1_size_used = anchor1_size
                                        statement.anchor2_color_used = anchor2_color
                                        statement.anchor2_size_used = anchor2_size

                                        if (self.objects[anchor1]['nyu_label']==self.objects[anchor2]['nyu_label'] and not
                                        (len(anchor1_color) > 0 or len(anchor1_size) > 0 or len(anchor2_color) > 0 or len(anchor2_size) > 0)):
                                            statement.sentence = aux_sentence
                                        elif (self.objects[target]['nyu_label']==self.objects[anchor1]['nyu_label'] and not
                                        (len(target_color) > 0 or len(target_size) > 0 or len(anchor1_color) > 0 or len(anchor1_size) > 0)):
                                            statement.replace("%other1%", "other ")
                                        elif (self.objects[target]['nyu_label']==self.objects[anchor2]['nyu_label'] and not
                                        (len(target_color) > 0 or len(target_size) > 0 or len(anchor2_color) > 0 or len(anchor2_size) > 0)):
                                            statement.replace("%other2%", "other ")


                                        statement.replace("%other1%", "")
                                        statement.replace("%other2%", "")
                                        statement.replace("%target_color%", target_color)
                                        statement.replace("%anchor1_color%", anchor1_color)
                                        statement.replace("%anchor2_color%", anchor2_color)
                                        statement.replace("%target_size%", target_size)
                                        statement.replace("%anchor1_size%", anchor1_size)
                                        statement.replace("%anchor2_size%", anchor2_size)
                                        statement.replace('%relation%', relation)

                                        statement_candidates.append(statement)

        return statement_candidates



    def condition_check(self, target, anchors, condition):
        """
        Finds uniquely identifiable objects and combinatorially generates relational statements
            Params:
                target(int): Index of target object
                anchor(int):Index of anchor object
                conditions(string): Set of conditions defined for this relation
            Returns:
                condition_check(boolean): Whether the target and anchor follow  the conditions
        """
        if self.objects[target]['nyu_label'] == 'ceiling' or self.objects[target]['nyu_label'] == 'floor':
            return False
        if self.objects[anchors[0]]['nyu_label'] == 'ceiling' or self.objects[anchors[1]] == 'ceiling':
            return False
        if self.objects[anchors[0]]['nyu_label'] == 'floor' or self.objects[anchors[0]]['nyu_label'] == 'floor':
            return False
        return True

