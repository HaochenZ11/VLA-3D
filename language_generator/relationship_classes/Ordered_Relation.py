from relationship_classes.Relationship import Relationship
from object_filtering import ObjectFilter
from Statement import Statement
import json
import time


class Ordered(Relationship):
    '''
    Class for Above object relationships in scene
    '''
    def __init__(self, region_graph, relation_language_template, objects, language_master_template, object_filter):
        """
        Parses arguments from command line for input scene graph and language template
           Params:
               region_graph(Dict): Scene data from a region
               relation_langauge_template(Dict): Template of language syntax and relational sentence structures
               language_master_template(Dict): Template of language syntax and object level grammar structures
               object_filter(ObjectFilter): ObjectFilter object with the scene object data
        """

        super().__init__(region_graph, objects, language_master_template, object_filter)

        self.relation_language_template = relation_language_template
        self.relation = self.relation_language_template['relation']
        if self.relation in self.region_graph:
            self.target_anchors = self.region_graph[self.relation]
        else:
            self.target_anchors = {}

        self.relation_type = 'ordered'


    def generate_statements(self, generation_configs, max_statements=None):
        """
        Generates dataset of above relational statements with ground truth attributes of the objects
            Params:
                max_statements(int): Maximum amount of statements per relation
            Returns:
                statements(Dict): Dictionary of relational statement and corresponding ground truth data
        """
        statements = {}

        template_sets = self.relation_language_template['templates']


        for t_set in template_sets:
            conditions = t_set["conditions"]

            for anchor_class, anchors_ids in self.target_anchors.items():
                if anchor_class == 'space' or anchor_class in generation_configs['exclude_classes']:
                    continue

                for anchor_id, targets in anchors_ids.items():

                    for target_class, target_ids in targets.items():
                        if target_class in generation_configs['exclude_classes'] or (target_class == 'space' and not generation_configs['include_spaces']):
                            continue

                        for target_id in target_ids:
                            distractors = target_ids.copy()
                            distractors.remove(target_id)

                            if self.condition_check(target_class, anchor_class, conditions):
                                statement_candidates = self.get_statement_candidates(t_set, target_id, anchor_id, target_class, anchor_class)

                                for statement in statement_candidates:
                                    if statement.sentence not in statements:
                                        statements[statement.sentence] = []

                                    target_pos = self.objects[target_id]["center"]
                                    anchor_pos = self.objects[anchor_id]["center"]
                                    target_color = self.objects[target_id]["color_labels"]
                                    anchor_color = self.objects[anchor_id]["color_labels"]
                                    target_size = self.objects[target_id]["volume"]
                                    anchor_size = self.objects[anchor_id]["volume"]

                                    statements[statement.sentence].append({"target_index": target_id, "anchor_index": anchor_id,
                                                          "target_class": target_class, "anchor_class": anchor_class,
                                                          # "target_nyu_id": target_nyu_id, "anchor_nyu_id": anchor_nyu_id,
                                                          "target_position": target_pos, "anchor_position": anchor_pos,
                                                          "target_colors": target_color, "anchor_colors": anchor_color,
                                                          "target_size": target_size, "anchor_size": anchor_size,
                                                          "target_color_used": statement.target_color_used, "target_size_used": statement.target_size_used,
                                                          "anchor_color_used": statement.anchor_color_used, "anchor_size_used": statement.anchor_size_used,
                                                          "distractor_ids": distractors, "relation": self.relation,
                                                          "relation_type": self.relation_type})

            if max_statements is not None and len(statements) == max_statements:
                return statements

        return statements



    def get_statement_candidates(self, template_set, target, anchor, target_class, anchor_class):
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

        for sentence in template_set["sentences"]:
            if target_class in self.language_master_template['verb_index']:
                target_verb_index = self.language_master_template['verb_index'][target_class]
            else:
                target_verb_index = 0
            target_verb = self.language_master_template['verbs'][target_verb_index]

            sentence = sentence.replace('%target%', target_class)
            sentence = sentence.replace('%target_verb%', target_verb)
            sentence = sentence.replace('%anchor%', anchor_class)

            object_filter = self.object_filter


            #INTERCLASS ATTRIBUTE FILTERING
            order, anchor_colors, anchor_sizes = object_filter.filter_targets_and_anchors_ordered(target, anchor, target_class, anchor_class, self.relation)

            if order is not None:
                order = self.relation_language_template["order"][order]

            for anchor_color in anchor_colors:
                for anchor_size in anchor_sizes:
                    for relation in relation_synonyms:
                        for o in order:

                            statement = Statement(sentence, self.relation, self.relation_type)

                            statement.anchor_color_used = anchor_color
                            statement.anchor_size_used = anchor_size
                            statement.target_color_used = ""
                            statement.target_size_used = ""

                            if (target_class == anchor_class):
                                continue

                            statement.replace("%other%", "")
                            statement.replace("%target_color%", "")
                            statement.replace("%anchor_color%", anchor_color)
                            statement.replace("%target_size%", "")
                            statement.replace("%anchor_size%", anchor_size)
                            statement.replace('%relation%', relation)
                            statement.replace("%target_order%", o)

                            statement_candidates.append(statement)

        return statement_candidates

    def condition_check(self, target_class, anchor_class, condition):
        """
        Finds uniquely identifiable objects and combinatorially generates relational statements
            Params:
                target(int): Index of target object
                anchor(int):Index of anchor object
                conditions(string): Set of conditions defined for this relation
            Returns:
                condition_check(boolean): Whether the target and anchor follow  the conditions
        """
        if target_class == 'ceiling' or target_class == 'floor':
            return False
        if anchor_class == 'ceiling' or anchor_class == 'floor':
            return False
        return True

