from collections import namedtuple

#CSP are always the first two odors

class PIR:
    condition = 'PIR'
    paths = ['E:/IMPORTANT DATA/DATA_2P/M183/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M184/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M199_pir/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M200_pir/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M201_pir/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M202_pir/training_LEARNING']
    odors = [
        ['pin', 'msy', '4mt', 'lim'],
        ['pin', 'msy', '4mt', 'lim'],
        ['euy', 'lim', 'iso', '4mt'],
        ['euy', 'lim', 'iso', '4mt'],
        ['euy', 'lim', 'fen', 'ger'],
        ['euy', 'lim', 'iso', '4mt'],
    ]
    csp = [['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy'],
           ]

class PIR_NAIVE:
    condition = 'PIR_NAIVE'
    pass

class OFC:
    #
    condition = 'OFC'
    paths = ['E:/IMPORTANT DATA/DATA_2P/M187_ofc/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M188_ofc/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M206_ofc/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M233_ofc/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M234_ofc/training_LEARNING']
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy'],
           ]

class OFC_STATE:
    pass

class OFC_CONTEXT:
    pass

class OFC_REVERSAL:
    pass

class OFC_COMPOSITE:
    condition = 'OFC_COMPOSITE'
    paths = ['E:/IMPORTANT DATA/DATA_2P/M2_OFC/training', #days 1-4
             'E:/IMPORTANT DATA/DATA_2P/M3_OFC/training', #days 1-4, odor currently nananaive
             'E:/IMPORTANT DATA/DATA_2P/M4_OFC/training', #days 1-6, odor currently nananaive
             'E:/IMPORTANT DATA/DATA_2P/M5_OFC/training'] #days 1-4, odor currently nanaive]
    odors = [
        ['naive', 'oct'],
        ['naive', 'oct'],
        ['naive', 'oct'],
        ['naive', 'oct']
    ]
    csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct']
    ]

class OFC_JAWS:
    pass

class MPFC_COMPOSITE:
    condition = 'MPFC_COMPOSITE'
    paths = ['E:/IMPORTANT DATA/DATA_2P/M2_MPFC/training',  #days 1-3
             'E:/IMPORTANT DATA/DATA_2P/M3_MPFC/training',  #days 1-3
             'E:/IMPORTANT DATA/DATA_2P/M4_MPFC/training',  #days 1-4, naive first day
             'E:/IMPORTANT DATA/DATA_2P/M6_MPFC/training']  #days 1-4, nanaive first day.]
    odors = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct']
    ]


    pass

class BLA_JAWS:
    pass

class BLA:
    #TODO: N and T need fixing
    condition = 'BLA'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M211_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M212_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M213_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M214_bla/training_LEARNING',
        ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy']
           ]

class BLA_STATE:
    pass

class BLA_CONTEXT:
    pass

class BLA_REVERSAL:
    pass





class BLA_LT:
    pass



