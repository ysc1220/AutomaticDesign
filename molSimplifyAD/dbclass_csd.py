import os
import getpass
import numpy as np
from datetime import datetime

mongo_attr_other = ["date", "author", "tag"]
mongo_attr_inherent = ["refcode", "refcode_plus", "geometry", "mol2_string", "has_csd_user_charges",
                       "e_counts", "odd_even_e_counts", "metal", "charge"]  ### check the names with Michael!!
mongo_attr_optional = ["charge", "spin", "ox"]
convert_dict = {"charge": "complex_formal_charge_csd", }


class CSDMongo():
    '''
    Class that parses useful information of a CSD complex in order to run a calculation.
    Note that this class is designed to interface with CSD database (import CSD complexes to db.csd collection).
    Results of DFT calculations are still restored in db.oct collection by tmcMongo class.

    '''

    def __init__(self, csdobj, tag=False, update_fields=False):
        self.csdobj = csdobj
        self.tag = tag
        self.date = datetime.now()
        self.author = getpass.getuser()
        self.document = {}
        if update_fields:
            self.update_fields = update_fields
        else:
            self.update_fields = list()
        self.construct_documents()

    def construct_documents(self):
        for attr in mongo_attr_other:
            self.document.update({attr: getattr(self, attr)})
        for attr in mongo_attr_inherent:
            attr_csd = attr
            if attr in list(convert_dict.keys()):
                attr_csd = convert_dict[attr]
            if attr_csd in self.csdobj.__dict__:
                if not attr_csd == "graphs":
                    self.document.update({attr: getattr(self.csdobj, attr_csd)})
                else:
                    self.document.update({attr: getattr(self.csdobj, attr_csd).tolist()})  ## for 2d np.array
            else:
                raise ValueError("Attribute %s missing from csdobj input." % attr)
        for attr in mongo_attr_optional:
            attr_csd = attr
            if attr in list(convert_dict.keys()):
                attr_csd = convert_dict[attr]
            if attr_csd in self.csdobj.__dict__:
                self.document.update({attr: getattr(self.csdobj, attr_csd)})
