import json
from Statement import Statement
import random
import re

class ObjectFilter:
    '''
    Class for ObjectFilter object responsible for filtering and ranking objects in scene by their attributes
    '''
    def __init__(self, spatial_relations_by_anchor, spatial_relations_by_target, objects, objects_class_region, metadata):
        """
        All children od the Relationship class must generate statements
            Params:
               region_graph(Dict): Scene data from a region
        """
        self.spatial_relations_by_anchor = spatial_relations_by_anchor
        self.spatial_relations_by_target = spatial_relations_by_target
        self.objects = objects
        self.objects_class_region = objects_class_region
        with open(metadata) as c:
            self.metadata = json.load(c)
        self.all_colors = set(self.metadata['colors_used'])
        self.all_objects = set(self.metadata['object_set'])
        self.all_target_classes = self.metadata['target_objects_by_relation']
        self.all_anchor_classes = self.metadata['anchor_objects_by_relation']

    def __find_all_occurrences__(self, text, substring):
        positions = []
        pos = text.find(substring)

        while pos != -1:
            positions.append(pos)
            pos = text.find(substring, pos + len(substring))

        return positions

    def __replace_range__(self, text, i, j, replacement):
        # Ensure i and j are within the valid range
        if i < 0:
            i = 0
        if j > len(text):
            j = len(text)
        if i > j:
            raise ValueError("Start index cannot be greater than end index")

        # Replace the substring from i to j with the replacement string
        return text[:i] + replacement + text[j:]


    def get_distractors(self, object_id, object_class):
        distractors = self.objects_class_region[object_class].copy()
        distractors.remove(object_id)
        return distractors

    def get_false_statements(
            self,
            statement,
            target_class,
            anchor_class,
            relation
    ):
        false_statements = {}

        unavailable_target_colors = set()
        unavailable_target_classes = set()
        unavailable_anchor_colors = set()
        unavailable_anchor_classes = set()

        for a_id, targets in self.spatial_relations_by_anchor[relation][anchor_class].items():
            for t_class, t_ids in targets.items():
                unavailable_target_classes.add(t_class)
                for t_id in t_ids:
                    unavailable_target_colors.update(self.objects[t_id]['color_labels'])

        for t_id, anchors in self.spatial_relations_by_target[relation][target_class].items():
            for a_class, a_ids in anchors.items():
                unavailable_anchor_classes.add(a_class)
                for a_id in a_ids:
                    unavailable_anchor_colors.update(self.objects[a_id]['color_labels'])

        available_target_colors = list(self.all_colors - unavailable_target_colors)
        available_target_classes = list(set(self.all_target_classes[relation]) - unavailable_target_classes)
        available_anchor_colors = list(self.all_colors - unavailable_anchor_colors)
        available_anchor_classes = list(set(self.all_anchor_classes[relation]) - unavailable_anchor_classes)

        false_target_color = None
        if len(available_target_colors) > 0:
            false_target_color = random.choice(available_target_colors)

        false_target_class = None
        if len(available_target_classes) > 0:
            false_target_class = random.choice(available_target_classes)

        false_anchor_color = None
        if len(available_anchor_colors) > 0:
            false_anchor_color = random.choice(available_anchor_colors)

        false_anchor_class = None
        if len(available_anchor_classes) > 0:
            false_anchor_class = random.choice(available_anchor_classes)

        if false_target_color is not None:
            if statement.target_color_used != "":
                occurrences = self.__find_all_occurrences__(statement.sentence, statement.target_color_used)
                if len(occurrences) > 1:
                    false_statements['false_target_color'] = self.__replace_range__(statement.sentence, occurrences[0], occurrences[0] + len(statement.target_color_used), f"{false_target_color}")
                else:
                    false_statements['false_target_color'] = statement.sentence.replace(statement.target_color_used, f"{false_target_color}")

            else:
                occurrences = self.__find_all_occurrences__(statement.sentence, target_class)
                if len(occurrences) > 1:
                    false_statements['false_target_color'] = self.__replace_range__(statement.sentence, occurrences[0], occurrences[0] + len(target_class), f"{false_target_color} {target_class}")
                else:
                    false_statements['false_target_color'] = statement.sentence.replace(target_class, f"{false_target_color} {target_class}")

        if false_target_class is not None:
            occurrences = self.__find_all_occurrences__(statement.sentence, target_class)
            if len(occurrences) > 1:
                false_statements['false_target_class'] = self.__replace_range__(statement.sentence, occurrences[0], occurrences[0] + len(target_class), f"{false_target_class}")
            else:
                false_statements['false_target_class'] = statement.sentence.replace(target_class, false_target_class)

        false_statements['false_anchors'] = {"anchor1": {}}

        if false_anchor_color is not None:
            if statement.anchor_color_used != "":
                occurrences = self.__find_all_occurrences__(statement.sentence, statement.anchor_color_used)
                if len(occurrences) > 1:
                    false_statements['false_anchors']["anchor1"]['false_anchor_color'] = self.__replace_range__(statement.sentence, occurrences[1], occurrences[1] + len(statement.anchor_color_used), f"{false_anchor_color}")
                else:
                    false_statements['false_anchors']["anchor1"]['false_anchor_color'] = statement.sentence.replace(statement.anchor_color_used, f"{false_anchor_color}")

            else:
                occurrences = self.__find_all_occurrences__(statement.sentence, anchor_class)
                if len(occurrences) > 1:
                    false_statements['false_anchors']["anchor1"]['false_anchor_color'] = self.__replace_range__(statement.sentence, occurrences[1], occurrences[1] + len(anchor_class), f"{false_anchor_color} {anchor_class}")
                else:
                    false_statements['false_anchors']["anchor1"]['false_anchor_color'] = statement.sentence.replace(anchor_class, f"{false_anchor_color} {anchor_class}")

        if false_anchor_class is not None:
            occurrences = self.__find_all_occurrences__(statement.sentence, anchor_class)
            if len(occurrences) > 1:
                false_statements['false_anchors']["anchor1"]['false_anchor_class'] = self.__replace_range__(statement.sentence, occurrences[1], occurrences[1] + len(anchor_class), f"{false_anchor_class}")
            else:
                false_statements['false_anchors']["anchor1"]['false_anchor_class'] = statement.sentence.replace(anchor_class, false_anchor_class)

        return false_statements


    def get_false_statements_ternary(
            self,
            statement,
            target_class,
            anchor1_id,
            anchor2_id,
            relation,
    ):

        false_statements = {}
        target_anchors = self.spatial_relations_by_anchor[relation]
        unavailable_target_colors = set()
        unavailable_target_classes = set()
        unavailable_anchor_classes = set()
        unavailable_anchor_colors = set()

        for t_class, target_anchors in target_anchors.items():
            for t_id, anchors in target_anchors.items():
                for anchor_set in anchors:
                    if anchor_set == [anchor1_id, anchor2_id] or anchor_set == [anchor2_id, anchor1_id]:
                        unavailable_target_classes.add(t_class)
                        unavailable_target_colors.update(self.objects[t_id]['color_labels'])
                    if t_class == target_class:
                        unavailable_anchor_classes.update([self.objects[anchor1_id]['nyu_label'], self.objects[anchor2_id]['nyu_label']])
                        unavailable_anchor_colors.update(self.objects[anchor1_id]['color_labels'])


        available_target_colors = list(self.all_colors - unavailable_target_colors)
        available_target_classes = list(set(self.all_target_classes[relation]) - unavailable_target_classes)
        available_anchor_colors = list(self.all_colors - unavailable_anchor_colors)
        available_anchor_classes = list(set(self.all_anchor_classes[relation]) - unavailable_anchor_classes)

        false_target_color = None
        if len(available_target_colors) > 0:
            false_target_color = random.choice(available_target_colors)

        false_target_class = None
        if len(available_target_classes) > 0:
            false_target_class = random.choice(available_target_classes)

        false_anchor1_color = None
        if len(available_anchor_colors) > 0:
            false_anchor1_color = random.choice(available_anchor_colors)

        false_anchor2_color = None
        if len(available_anchor_colors) > 0:
            false_anchor2_color = random.choice(available_anchor_colors)

        false_anchor1_class = None
        if len(available_anchor_classes) > 0:
            false_anchor1_class = random.choice(available_anchor_classes)

        available_anchor_classes.remove(false_anchor1_class)
        false_anchor2_class = None
        if len(available_anchor_classes) > 0:
            false_anchor2_class = random.choice(available_anchor_classes)

        if false_target_color is not None:
            if statement.target_color_used != "":
                occurrences = self.__find_all_occurrences__(statement.sentence, statement.target_color_used)
                if len(occurrences) > 1:
                    false_statements['false_target_color'] = self.__replace_range__(statement.sentence, occurrences[0], occurrences[0] + len(statement.target_color_used), f"{false_target_color}")
                else:
                    false_statements['false_target_color'] = statement.sentence.replace(statement.target_color_used, f"{false_target_color}")
            else:
                occurrences = self.__find_all_occurrences__(statement.sentence, target_class)
                if len(occurrences) > 1:
                    false_statements['false_target_color'] = self.__replace_range__(statement.sentence, occurrences[0], occurrences[0] + len(target_class), f"{false_target_color} {target_class}")
                else:
                    false_statements['false_target_color'] = statement.sentence.replace(target_class, f"{false_target_color} {target_class}")

        if false_target_class is not None:
            occurrences = self.__find_all_occurrences__(statement.sentence, target_class)
            if len(occurrences) > 1:
                false_statements['false_target_class'] = self.__replace_range__(statement.sentence, occurrences[0], occurrences[0] + len(target_class), f"{false_target_class}")
            else:
                false_statements['false_target_class'] = statement.sentence.replace(target_class, false_target_color)

            false_statements['false_anchors'] = {"anchor1": {}, "anchor2": {}}

        if false_anchor1_color is not None:
            if statement.anchor1_color_used != "":
                occurrences = self.__find_all_occurrences__(statement.sentence, statement.anchor1_color_used)
                if len(occurrences) == 3 or len(occurrences) == 2 and statement.anchor1_color_used == statement.target_color_used:
                    false_statements['false_anchors']["anchor1"]['false_anchor_color'] = self.__replace_range__(statement.sentence, occurrences[1], occurrences[1] + len(statement.anchor1_color_used), f"{false_anchor1_color}")
                else:
                    false_statements['false_anchors']["anchor1"]['false_anchor_color'] = statement.sentence.replace(statement.anchor1_color_used, f"{false_anchor1_color}")
            else:
                occurrences = self.__find_all_occurrences__(statement.sentence, self.objects[anchor1_id]['nyu_label'])
                if len(occurrences) > 1:
                    false_statements['false_anchors']["anchor1"]['false_anchor_color'] = self.__replace_range__(statement.sentence, occurrences[1], occurrences[1] + len(self.objects[anchor1_id]['nyu_label']), f"{false_anchor1_color} {self.objects[anchor1_id]['nyu_label']}")
                else:
                    false_statements['false_anchors']["anchor1"]['false_anchor_color'] = statement.sentence.replace(self.objects[anchor1_id]['nyu_label'], f"{false_anchor1_color} {self.objects[anchor1_id]['nyu_label']}")

        if false_anchor1_class is not None:
            occurrences = self.__find_all_occurrences__(statement.sentence, self.objects[anchor1_id]['nyu_label'])
            if len(occurrences) > 1:
                false_statements['false_anchors']["anchor1"]['false_anchor_class'] = self.__replace_range__(statement.sentence, occurrences[1], occurrences[1] + len(self.objects[anchor1_id]['nyu_label']), f"{false_anchor1_class}")
            else:
                false_statements['false_anchors']["anchor1"]['false_anchor_class'] = statement.sentence.replace(self.objects[anchor1_id]['nyu_label'], false_anchor1_class)

        if false_anchor2_color is not None:
            if statement.anchor2_color_used != "":
                occurrences = self.__find_all_occurrences__(statement.sentence, statement.anchor2_color_used)
                if len(occurrences) == 3:
                    false_statements['false_anchors']["anchor2"]['false_anchor_color'] = self.__replace_range__(statement.sentence, occurrences[2], occurrences[2] + len(statement.anchor2_color_used), f"{false_anchor2_color}")
                elif len(occurrences) == 2 and statement.anchor2_color_used == statement.target_color_used:
                    false_statements['false_anchors']["anchor2"]['false_anchor_color'] = self.__replace_range__(statement.sentence, occurrences[1], occurrences[1] + len(statement.anchor2_color_used), f"{false_anchor2_color}")
                else:
                    false_statements['false_anchors']["anchor2"]['false_anchor_color'] = statement.sentence.replace(statement.anchor2_color_used, f"{false_anchor2_color}")
            else:
                occurrences = self.__find_all_occurrences__(statement.sentence, self.objects[anchor2_id]['nyu_label'])
                if len(occurrences) > 1:
                    false_statements['false_anchors']["anchor2"]['false_anchor_color'] = self.__replace_range__(statement.sentence, occurrences[1], occurrences[1] + len(self.objects[anchor2_id]['nyu_label']), f"{false_anchor2_color} {self.objects[anchor2_id]['nyu_label']}")
                else:
                    false_statements['false_anchors']["anchor2"]['false_anchor_color'] = statement.sentence.replace(self.objects[anchor2_id]['nyu_label'], f"{false_anchor2_color} {self.objects[anchor2_id]['nyu_label']}")

        if false_anchor2_class is not None:
            occurrences = self.__find_all_occurrences__(statement.sentence, self.objects[anchor2_id]['nyu_label'])
            if len(occurrences) > 1:
                false_statements['false_anchors']["anchor2"]['false_anchor_class'] = self.__replace_range__(statement.sentence, occurrences[1], occurrences[1] + len(self.objects[anchor2_id]['nyu_label']), f"{false_anchor2_class}")
            else:
                false_statements['false_anchors']["anchor2"]['false_anchor_class'] = statement.sentence.replace(self.objects[anchor2_id]['nyu_label'], false_anchor2_class)

        return false_statements


    def filter_objects(self, object_id, object_list):

        other_same_objects = object_list.copy()
        other_same_objects.remove(object_id)

        object_color, object_size = [""], [""]

        if len(object_list) > 1:
            # Sort by color
            for i, color in enumerate(self.objects[object_id]['color_labels']):
                if color != 'N/A' and (float)(self.objects[object_id]['color_percentages'][i])> 0.25:
                    unique_color = True
                    for i in other_same_objects:
                        if color in self.objects[i]['color_labels']:
                            unique_color = False
                            # continue

                    if unique_color:
                        object_color, object_size = [color + " "], [""]
                        object_list = [object_id]
                        break

            # TODO: sort further by color
            if len(object_list) > 1:
                # Sort by size (largest surface area)
                sorted_size_list = sorted(object_list, key=lambda x: self.objects[x]['largest_face_area'])

                if sorted_size_list[0] == object_id:
                    if 1.2 * self.objects[sorted_size_list[0]]['largest_face_area'] < \
                            self.objects[sorted_size_list[1]]['largest_face_area']:
                        # TODO: relative size
                        object_color, object_size = [""], ["small "]
                        object_list = [object_id]

                if sorted_size_list[-1] == object_id:
                    if self.objects[sorted_size_list[-1]]['largest_face_area'] > 1.2 * \
                            self.objects[sorted_size_list[-2]]['largest_face_area']:
                        # TODO: relative size
                        object_color, object_size = [""], ["big "]
                        object_list = [object_id]

            if len(object_list) > 1:
                # Sort by dominant color and size
                for i, color in enumerate(self.objects[object_id]['color_labels']):
                    if color != 'N/A' and (float)(self.objects[object_id]['color_percentages'][i]) > 0.25:
                        other_same_color_objects = []
                        for i in other_same_objects:
                            if color in self.objects[i]['color_labels']:
                                other_same_color_objects.append(i)

                        colored_sorted_size_list = sorted(other_same_color_objects, key=lambda x: self.objects[x]['largest_face_area'])

                        if colored_sorted_size_list[0] == object_id:
                            if 1.2 * self.objects[colored_sorted_size_list[0]]['largest_face_area'] < \
                                    self.objects[colored_sorted_size_list[1]]['largest_face_area']:
                                # TODO: relative size
                                object_color, object_size = [color + " "], ["small "]
                                object_list = [object_id]
                                break

                        if colored_sorted_size_list[-1] == object_id:
                            if self.objects[colored_sorted_size_list[-1]]['largest_face_area'] > 1.2 * \
                                    self.objects[colored_sorted_size_list[-2]]['largest_face_area']:
                                # TODO: relative size
                                object_color, object_size = [color + " "], ["big "]
                                object_list = [object_id]
                                break

        return object_list, object_color, object_size


    def filter_targets_and_anchors(self, target, anchor, target_class, anchor_class, relation, filters=None):
        """
        All children of the Relationship class must generate statements
            Params:
                object(int): Index of the object to find unique qualifier in the region
                filters(List[string]): Order of importance for the attribute filters
            Returns:
                color(List): List of necessary color attributes to make object unique
                size(List): List of necessary size attributes to make object unique
        """
        target_anchors = self.spatial_relations_by_anchor[relation][anchor_class]

        filtered_target_anchors = {}

        for a_id, targets in target_anchors.items():
            for t_class, t_ids in targets.items():
                if t_class == target_class:
                    filtered_target_anchors[a_id] = t_ids

        target_list = filtered_target_anchors[anchor]
        anchor_list = self.objects_class_region[anchor_class]

        #Making anchor unique in the scene
        # anchor_list = list(filtered_target_anchors.keys())

        target_color, target_size, anchor_color, anchor_size = [""], [""], [""], [""]

        if len(target_list) == 1 and len(anchor_list) == 1:
            return target_color, target_size, anchor_color, anchor_size

        # ANCHOR FILTERING
        #TODO: Allow multi-color
        anchor_list, anchor_color, anchor_size = self.filter_objects(anchor, anchor_list)
        #Non-unique anchor
        if len(anchor_list) > 1:
            return [], [], [], []
        if len(target_list) == 1 and (len(anchor_list) == 1 or target_class == "space"):
            return target_color, target_size, anchor_color, anchor_size

        #TARGET FILTERING
        target_list, target_color, target_size = self.filter_objects(target, target_list)
        if len(target_list) == 1 and (len(anchor_list) == 1 or target_class == "space"):
            return target_color, target_size, anchor_color, anchor_size
        return [], [], [], []


    def filter_targets_and_anchors_ordered(self, target, anchor, target_class, anchor_class, relation, filters=None):
        """
        All children of the Relationship class must generate statements
            Params:
                object(int): Index of the object to find unique qualifier in the region
                filters(List[string]): Order of importance for the attribute filters
            Returns:
                color(List): List of necessary color attributes to make object unique
                size(List): List of necessary size attributes to make object unique
        """
        target_anchors = self.spatial_relations_by_anchor[relation][anchor_class]

        filtered_target_anchors = {}

        for a_id, targets in target_anchors.items():
            for t_class, t_ids in targets.items():
                if t_class == target_class:
                    filtered_target_anchors[a_id] = t_ids


        target_list = filtered_target_anchors[anchor]
        anchor_list = self.objects_class_region[anchor_class]

        #Making anchor unique in the scene
        # anchor_list = list(filtered_target_anchors.keys())

        if len(target_list) == 1:
            return None, [], []

        # ANCHOR FILTERING
        #TODO: Allow multi-color
        anchor_list, anchor_color, anchor_size = self.filter_objects(anchor, anchor_list)

        if len(anchor_list) > 1:
            return None, [], []

        order = target_list.index(target)
        if order > 2:
            return None, [], []
        order = str(order)

        return order, anchor_color, anchor_size



    def filter_targets_and_anchors_ternary(self, target, anchor1, anchor2, relation, filters=None):
        """
        All children of the Relationship class must generate statements
            Params:
                object(int): Index of the object to find unique qualifier in the region
                filters(List[string]): Order of importance for the attribute filters
            Returns:
                color(List): List of necessary color attributes to make object unique
                size(List): List of necessary size attributes to make object unique
        """

        target_object_name = self.objects[target]['nyu_label']
        anchor1_object_name = self.objects[anchor1]['nyu_label']
        anchor2_object_name = self.objects[anchor2]['nyu_label']

        target_anchors = self.spatial_relations_by_anchor[relation][target_object_name]

        filtered_target_anchors = {}

        for t_id, anchors in target_anchors.items():
            for anchor_set in anchors:
                if self.objects[anchor_set[0]]['nyu_label'] == anchor1_object_name and self.objects[anchor_set[1]]['nyu_label'] == anchor2_object_name:
                    if t_id not in filtered_target_anchors:
                        filtered_target_anchors[t_id] = []
                    filtered_target_anchors[t_id].append(anchor_set)


        target_list = list(filtered_target_anchors.keys())
        anchor1_list = self.objects_class_region[anchor1_object_name]
        anchor2_list = self.objects_class_region[anchor2_object_name]


        target_color, target_size, anchor1_color, anchor1_size, anchor2_color, anchor2_size = [""], [""], [""], [""], [""], [""]

        if len(target_list) == 1:
            return target_color, target_size, anchor1_color, anchor1_size, anchor2_color, anchor2_size

        # ANCHOR FILTERING
        anchor1_list = self.filter_objects(anchor1, anchor1_list)

        # Unique anchor
        if len(anchor1_list) > 1:
            return [], [], [], [], [], []

        anchor2_list = self.filter_objects(anchor2, anchor2_list)

        # Unique anchor
        if len(anchor2_list) > 1:
            return [], [], [], [], [], []

        #TARGET FILTERING
        target_list, target_color, target_size = self.filter_objects(target, target_list)

        if len(target_list) == 1:
            return target_color, target_size, anchor1_color, anchor1_size, anchor2_color, anchor2_size

        return [], [], [], [], [], []


    def get_objects(self):
        """
        All children od the Relationship class must generate statements
            Returns:
                Objects(List): List of object and their ground truth data from the region
        """
        return self.region_graph['objects']

    def get_object_names(self):
        """
        Getter function for object_names
            Returns:
                object_names(List): List of object names in index order
        """
        return self.object_names

    def get_object_counts(self):
        """
        Getter function for object_counts Counter
            Returns:
                object_counts(Counter): Counter object with object names
        """
        return self.object_counts
