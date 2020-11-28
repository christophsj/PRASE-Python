from objects.Entity import Entity
from objects.Relation import Relation
from objects.Attribute import Attribute
from objects.Literal import Literal


class KG:
    def __init__(self, name="KG"):
        self.name = name
        self.entity_set = set()
        self.relation_set = set()
        self.attribute_set = set()
        self.literal_set = set()

        self.entity_dict_by_name = dict()
        self.relation_dict_by_name = dict()
        self.attribute_dict_by_name = dict()
        self.literal_dict_by_name = dict()

        self.relation_tuple_list = list()
        self.attribute_tuple_list = list()

        self.relation_set_func_ranked = list()
        self.relation_set_func_inv_ranked = list()
        self.attribute_set_func_ranked = list()
        self.attribute_set_func_inv_ranked = list()

    def get_entity(self, name: str):
        if self.entity_dict_by_name.__contains__(name):
            return self.entity_dict_by_name.get(name)
        else:
            entity = Entity(idx=len(self.entity_set), name=name, affiliation=self)
            self.entity_set.add(entity)
            self.entity_dict_by_name[name] = entity
            return entity

    def get_relation(self, name: str):
        if self.relation_dict_by_name.__contains__(name):
            return self.relation_dict_by_name.get(name)
        else:
            relation = Relation(idx=len(self.relation_set), name=name, affiliation=self)
            self.relation_set.add(relation)
            self.relation_dict_by_name[name] = relation
            return relation

    def get_attribute(self, name: str):
        if self.attribute_dict_by_name.__contains__(name):
            return self.attribute_dict_by_name.get(name)
        else:
            attribute = Attribute(idx=len(self.attribute_set), name=name, affiliation=self)
            self.attribute_set.add(attribute)
            self.attribute_dict_by_name[name] = attribute
            return attribute

    def get_literal(self, name: str):
        if self.literal_dict_by_name.__contains__(name):
            return self.literal_dict_by_name.get(name)
        else:
            literal = Literal(name=name, affiliation=self)
            self.literal_set.add(literal)
            self.literal_dict_by_name[name] = literal
            return literal

    def insert_relation_tuple(self, head: str, relation: str, tail: str):
        ent_h, rel, ent_t = self.get_entity(head), self.get_relation(relation), self.get_entity(tail)
        ent_h.add_relation_as_head(relation=rel, tail=ent_t)
        rel.add_relation_tuple(head=ent_h, tail=ent_t)
        ent_t.add_relation_as_tail(relation=rel, head=ent_h)
        self.relation_tuple_list.append((ent_h, rel, ent_t))

    def insert_attribute_tuple(self, entity: str, attribute: str, literal: str):
        ent, attr, val = self.get_entity(entity), self.get_attribute(attribute), self.get_literal(literal)
        ent.add_attribute_tuple(attribute=attr, literal=val)
        attr.add_attribute_tuple(entity=ent, literal=val)
        val.add_attribute_tuple(entity=ent, attribute=attr)
        self.attribute_tuple_list.append((ent, attr, val))

    def calculate_functionality(self):
        for relation in self.relation_set:
            relation.calculate_functionality()
            self.relation_set_func_ranked.append(relation)
            self.relation_set_func_inv_ranked.append(relation)
        for attribute in self.attribute_set:
            attribute.calculate_functionality()
            self.attribute_set_func_ranked.append(attribute)
            self.attribute_set_func_inv_ranked.append(attribute)
        self.relation_set_func_ranked.sort(key=lambda x: x.functionality, reverse=True)
        self.relation_set_func_inv_ranked.sort(key=lambda x: x.functionality_inv, reverse=True)
        self.attribute_set_func_ranked.sort(key=lambda x: x.functionality, reverse=True)
        self.attribute_set_func_inv_ranked.sort(key=lambda x: x.functionality_inv, reverse=True)

    def print_kg_info(self):
        print("Information of Knowledge Graph (" + str(self.name) + "):")
        print("- Relation Tuple Number: " + str(len(self.relation_tuple_list)))
        print("- Attribute Tuple Number: " + str(len(self.attribute_tuple_list)))
        print("- Entity Number: " + str(len(self.entity_set)))
        print("- Relation Number: " + str(len(self.relation_set)))
        print("- Attribute Number: " + str(len(self.attribute_set)))
        print("- Literal Number: " + str(len(self.literal_set)))
        print("- Functionality Statistics:")
        print("--- TOP-10 Relations (Func) ---")
        for i in range(min(10, len(self.relation_set_func_ranked))):
            relation = self.relation_set_func_ranked[i]
            print("Name: " + relation.name + "\tFunc: " + str(relation.functionality))

        print("--- TOP-10 Relations (Func-Inv) ---")
        for i in range(min(10, len(self.relation_set_func_inv_ranked))):
            relation = self.relation_set_func_inv_ranked[i]
            print("Name: " + relation.name + "\tFunc-Inv: " + str(relation.functionality_inv))

        print("--- TOP-10 Attributes (Func) ---")
        for i in range(min(10, len(self.attribute_set_func_ranked))):
            attribute = self.attribute_set_func_ranked[i]
            print("Name: " + attribute.name + "\tFunc: " + str(attribute.functionality))

        print("--- TOP-10 Attributes (Func-Inv) ---")
        for i in range(min(10, len(self.attribute_set_func_inv_ranked))):
            attribute = self.attribute_set_func_inv_ranked[i]
            print("Name: " + attribute.name + "\tFunc-Inv: " + str(attribute.functionality_inv))
