class ObjectFilter:
    '''
    Class for ObjectFilter object responsible for filtering and ranking objects in scene by their attributes
    '''
    def __init__(self, spatial_relations, objects):
        """
        All children od the Relationship class must generate statements
            Params:
               region_graph(Dict): Scene data from a region
        """
        self.spatial_relations = spatial_relations
        self.objects = objects


    def get_distractors(self, target):
        object_name = self.objects['nyu_label'][target]
        object_list = self.objects[self.objects['nyu_label'] == object_name]
        other_same_objects = object_list.drop(target)
        return list(other_same_objects["object_id"])


    def filter_objects(self, object, filters=None):
        """
        All children of the Relationship class must generate statements
            Params:
                object(int): Index of the object to find unique qualifier in the region
                filters(List[string]): Order of importance for the attribute filters
            Returns:
                color(List): List of necessary color attributes to make object unique
                size(List): List of necessary size attributes to make object unique
        """
        object_name = self.objects['nyu_label'][object]
        object_list = self.objects[self.objects['nyu_label'] == object_name]

        color, size = [], []

        if len(object_list) == 1 or object_name == "space":
            return [""], [""]


        other_same_objects = object_list.drop(object)

        for color in self.objects['color_labels'][object]:
            if color != 'N/A':
                unique_color = True
                for other_colors in other_same_objects['color_labels']:
                    if color in other_colors:
                        unique_color = False
                if unique_color:
                    color, size = [color + " "], [""]
                    break


        object_list = object_list.sort_values(by=['size'])
        if object_list.iloc[0]['nyu_label'] == object_name and object_list.iloc[0]['center'] == self.objects['center'][object]:
            if 1.2 * object_list.iloc[0]['largest_face_area'] < object_list.iloc[1]['largest_face_area']:
                color, size = [""], ["small "]

        if object_list.iloc[-1]['nyu_label'] == object_name and object_list.iloc[-1]['center'] == self.objects['center'][object]:
            if object_list.iloc[-1]['largest_face_area'] > 1.2 * object_list.iloc[-2]['largest_face_area']:
                color, size = [""], ["big "]

        return color, size

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
        target_anchors = self.spatial_relations[relation][anchor_class]

        filtered_target_anchors = {}

        anchor_list = []
        for a_id, targets in target_anchors.items():
            anchor_list.append(a_id)
            for t_class, t_ids in targets.items():
                if t_class == target_class:
                    filtered_target_anchors[a_id] = t_ids

        target_list = filtered_target_anchors[anchor]

        #Making anchor unique in the scene
        # anchor_list = list(filtered_target_anchors.keys())

        target_color, target_size, anchor_color, anchor_size = [""], [""], [""], [""]

        if len(target_list) == 1 and len(anchor_list) == 1:
            return target_color, target_size, anchor_color, anchor_size

        other_same_targets = target_list.copy()
        other_same_targets.remove(target)

        other_same_anchors = anchor_list.copy()
        other_same_anchors.remove(anchor)


        # ANCHOR FILTERING
        #TODO: Allow multi-color
        if len(anchor_list) > 1:
            for color in self.objects[anchor]['color_labels']:
                if color != 'N/A':
                    unique_color = True
                    for i in other_same_anchors:
                        if color in self.objects[i]['color_labels']:
                            unique_color = False
                            # continue

                    if unique_color:
                        anchor_color, anchor_size = [color + " "], [""]
                        anchor_list = [anchor]
                        break

            #TODO: sort further by color
            if len(anchor_list) > 1:

                #Sort by largest surface area
                sorted_anchor_list = sorted(anchor_list, key=lambda x: self.objects[x]['largest_face_area'])

                if sorted_anchor_list[0] == anchor:
                    if 1.2 * self.objects[sorted_anchor_list[0]]['largest_face_area'] < self.objects[sorted_anchor_list[1]]['largest_face_area']:
                        #TODO: relative size
                        anchor_color, anchor_size = [""], ["small "]
                        anchor_list = [anchor]

                if sorted_anchor_list[-1] == anchor:
                    if self.objects[sorted_anchor_list[-1]]['largest_face_area'] > 1.2 * self.objects[sorted_anchor_list[-2]]['largest_face_area']:
                        #TODO: relative size
                        anchor_color, anchor_size = [""], ["big "]
                        anchor_list = [anchor]

        #Unique anchor
        if len(anchor_list) > 1:
            return [], [], [], []

        if len(target_list) == 1 and (len(anchor_list) == 1 or target_class == "space"):

            return target_color, target_size, anchor_color, anchor_size


        #TARGET FILTERING
        if len(target_list) > 1:
            for color in self.objects[target]['color_labels']:
                if color != 'N/A':
                    unique_color = True
                    for i in other_same_targets:
                        if color in self.objects[i]['color_labels']:
                            unique_color = False
                            # continue

                    if unique_color:
                        target_color, target_size = [color + " "], [""]
                        target_list = [target]
                        break

            #TODO: sort further by color
            if len(target_list) > 1:

                sorted_target_list = sorted(anchor_list, key=lambda x: self.objects[x]['largest_face_area'])

                if sorted_target_list[0] == target:
                    if 1.2 * self.objects[sorted_target_list[0]]['largest_face_area'] < self.objects[sorted_target_list[1]]['largest_face_area']:
                        target_color, target_size = [""], ["small "]
                        target_list = [target]

                if sorted_target_list[-1] == target:
                    if self.objects[sorted_target_list[-1]]['largest_face_area'] > 1.2 * self.objects[sorted_target_list[-2]]['largest_face_area']:
                        target_color, target_size = [""], ["big "]
                        target_list = [target]

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
        target_anchors = self.spatial_relations[relation][anchor_class]

        filtered_target_anchors = {}

        anchor_list = []
        for a_id, targets in target_anchors.items():
            anchor_list.append(a_id)
            for t_class, t_ids in targets.items():
                if t_class == target_class:
                    filtered_target_anchors[a_id] = t_ids


        target_list = filtered_target_anchors[anchor]

        #Making anchor unique in the scene
        # anchor_list = list(filtered_target_anchors.keys())

        order, anchor_color, anchor_size = "", [""], [""]

        if len(target_list) == 1:

            return None, [], []

        other_same_anchors = anchor_list.copy()
        other_same_anchors.remove(anchor)


        # ANCHOR FILTERING
        #TODO: Allow multi-color
        if len(anchor_list) > 1:
            for color in self.objects[anchor]['color_labels']:
                if color != 'N/A':
                    unique_color = True
                    for i in other_same_anchors:
                        if color in self.objects[i]['color_labels']:
                            unique_color = False
                            # continue

                    if unique_color:
                        anchor_color, anchor_size = [color + " "], [""]
                        anchor_list = [anchor]
                        break

            #TODO: sort further by color
            if len(anchor_list) > 1:
                sorted_anchor_list = sorted(anchor_list, key=lambda x: self.objects[x]['largest_face_area'])

                if sorted_anchor_list[0] == anchor:
                    if 1.2 * self.objects[sorted_anchor_list[0]]['largest_face_area'] < self.objects[sorted_anchor_list[1]]['largest_face_area']:
                        #TODO: relative size
                        anchor_color, anchor_size = [""], ["small "]
                        anchor_list = [anchor]

                if sorted_anchor_list[-1] == anchor:
                    if self.objects[sorted_anchor_list[-1]]['largest_face_area'] > 1.2 * self.objects[sorted_anchor_list[-2]]['largest_face_area']:
                        #TODO: relative size
                        anchor_color, anchor_size = [""], ["big "]
                        anchor_list = [anchor]


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

        target_anchors = self.spatial_relations[relation][target_object_name]

        filtered_target_anchors = {}

        for t_id, anchors in target_anchors.items():
            for anchor_set in anchors:
                if self.objects[anchor_set[0]]['nyu_label'] == anchor1_object_name and self.objects[anchor_set[1]]['nyu_label'] == anchor2_object_name:
                    if t_id not in filtered_target_anchors:
                        filtered_target_anchors[t_id] = []
                    filtered_target_anchors[t_id].append(anchor_set)


        target_list = list(filtered_target_anchors.keys())
        anchor1_list = list(self.objects.keys())
        anchor2_list = list(self.objects.keys())



        target_color, target_size, anchor1_color, anchor1_size, anchor2_color, anchor2_size = [""], [""], [""], [""], [""], [""]

        if len(target_list) == 1:
            return target_color, target_size, anchor1_color, anchor1_size, anchor2_color, anchor2_size

        other_same_targets = target_list.copy()
        other_same_targets.remove(target)
        other_same_anchor1 = anchor1_list.copy()
        other_same_anchor1.remove(anchor1)
        other_same_anchor2 = anchor2_list.copy()
        other_same_anchor2.remove(anchor2)



        #ANCHO1 FILTERING
        if len(anchor1_list) > 1:
            for color in self.objects[anchor1]['color_labels']:
                if color != 'N/A':
                    unique_color = True
                    for i in other_same_anchor1:
                        if color in self.objects[i]['color_labels']:
                            unique_color = False
                            # continue

                    if unique_color:
                        anchor1_color, anchor1_size = [color + " "], [""]
                        anchor1_list = [anchor1]
                        break

            #TODO: sort further by color
            if len(anchor1_list) > 1:

                #Sort by largest surface area
                sorted_anchor1_list = sorted(anchor1_list, key=lambda x: self.objects[x]['largest_face_area'])

                if sorted_anchor1_list[0] == anchor1:
                    if 1.2 * self.objects[sorted_anchor1_list[0]]['largest_face_area'] < self.objects[sorted_anchor1_list[1]]['largest_face_area']:
                        #TODO: relative size
                        anchor1_color, anchor1_size = [""], ["small "]
                        anchor1_list = [anchor1]

                if sorted_anchor1_list[-1] == anchor1:
                    if self.objects[sorted_anchor1_list[-1]]['largest_face_area'] > 1.2 * self.objects[sorted_anchor1_list[-2]]['largest_face_area']:
                        #TODO: relative size
                        anchor1_color, anchor1_size = [""], ["big "]
                        anchor1_list = [anchor1]

        # Unique anchor
        if len(anchor1_list) > 1:
            return [], [], [], [], [], []




        if len(anchor2_list) > 1:
            for color in self.objects[anchor2]['color_labels']:
                if color != 'N/A':
                    unique_color = True
                    for i in other_same_anchor2:
                        if color in self.objects[i]['color_labels']:
                            unique_color = False
                            # continue

                    if unique_color:
                        anchor2_color, anchor2_size = [color + " "], [""]
                        anchor2_list = [anchor2]
                        break

            #TODO: sort further by color
            if len(anchor2_list) > 1:

                #Sort by largest surface area
                sorted_anchor2_list = sorted(anchor2_list, key=lambda x: self.objects[x]['largest_face_area'])

                if sorted_anchor2_list[0] == anchor2:
                    if 1.2 * self.objects[sorted_anchor2_list[0]]['largest_face_area'] < self.objects[sorted_anchor2_list[1]]['largest_face_area']:
                        #TODO: relative size
                        anchor2_color, anchor2_size = [""], ["small "]
                        anchor2_list = [anchor2]

                if sorted_anchor2_list[-1] == anchor2:
                    if self.objects[sorted_anchor2_list[-1]]['largest_face_area'] > 1.2 * self.objects[sorted_anchor2_list[-2]]['largest_face_area']:
                        #TODO: relative size
                        anchor2_color, anchor2_size = [""], ["big "]
                        anchor2_list = [anchor2]

        # Unique anchor
        if len(anchor2_list) > 1:
            return [], [], [], [], [], []



        #TARGET FILTERING
        if len(target_list) > 1:
            for color in self.objects[target]['color_labels']:
                if color != 'N/A':
                    unique_color = True
                    for i in other_same_targets:
                        if color in self.objects[i]['color_labels']:
                            unique_color = False
                            # continue

                    if unique_color:
                        target_color, target_size = [color + " "], [""]
                        target_list = [target]
                        break
            #TODO: sort further by color
            if len(target_list) > 1:
                sorted_target_list = sorted(target_list, key=lambda x: self.objects[x]['largest_face_area'])

                if sorted_target_list[0] == target:
                    if 1.2 * self.objects[sorted_target_list[0]]['largest_face_area'] < self.objects[sorted_target_list[1]]['largest_face_area']:
                        target_color, target_size = [""], ["small "]
                        target_list = [target]

                if sorted_target_list[-1] == target:
                    if self.objects[sorted_target_list[-1]]['largest_face_area'] > 1.2 * self.objects[sorted_target_list[-2]]['largest_face_area']:
                        target_color, target_size = [""], ["big "]
                        target_list = [target]

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
