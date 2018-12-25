from collections import namedtuple

#CSP are always the first two odors

def all_conditions():
    out = [
        PIR, PIR_NAIVE,
        OFC, OFC_COMPOSITE,
        BLA,
        MPFC_COMPOSITE
    ]
    return out

class PIR:
    name = 'PIR'
    paths = ['E:/IMPORTANT DATA/DATA_2P/M183/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M184/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M199_pir/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M200_pir/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M201_pir/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M202_pir/training_LEARNING']
    odors = [
        ['pin', 'msy', '4mt', 'lim'],
        ['pin', 'msy', '4mt', 'lim'],
        ['euy', 'lim', 'iso', '4mt'], #pin
        ['euy', 'lim', 'iso', '4mt'], #pin
        ['euy', 'lim', 'fen', 'ger'],
        ['euy', 'lim', 'iso', '4mt'], #pin
    ]
    csp = [['pin', 'msy'],
           ['pin', 'msy'],
           ['euy', 'lim'],
           ['euy', 'lim'],
           ['euy', 'lim'],
           ['euy', 'lim']
           ]
    timing_override = [True, True, True, True, True, True]

class PIR_NAIVE:
    name = 'PIR_NAIVE'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M199_pir/training_NAIVE',
        'E:/IMPORTANT DATA/DATA_2P/M200_pir/training_NAIVE',
        'E:/IMPORTANT DATA/DATA_2P/M201_pir/training_NAIVE',
        'E:/IMPORTANT DATA/DATA_2P/M202_pir/training_NAIVE'
             ]
    odors = [
        ['2pe', 'ben', 'msy', 'oct'],
        ['2pe', 'ben', 'msy', 'oct'],
        ['2pe', 'ben', 'msy', 'oct'],
        ['2pe', 'ben', 'msy', 'oct']
    ]
    timing_override = [True, True, True, True]


class OFC:
    name = 'OFC'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M187_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M188_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M206_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M233_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M234_ofc/training_LEARNING'
    ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
           ]
    timing_override = [False, False, False, True, True]

class OFC_STATE:
    pass

class OFC_CONTEXT:
    pass

class OFC_REVERSAL:
    pass

class OFC_LONGTERM:
    #TODO: exclude additional odors
    name = 'OFC_LONGTERM'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M230_ofc/training_LONGTERM',
        'E:/IMPORTANT DATA/DATA_2P/M232_ofc/training_LONGTERM',
        'E:/IMPORTANT DATA/DATA_2P/M239_ofc/training_LONGTERM',
        'E:/IMPORTANT DATA/DATA_2P/M241_ofc/training_LONGTERM'
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
    timing_override = [True, True, True, True]

class OFC_JAWS:
    name = 'OFC_JAWS'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M222_jawsofc_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M226_jawsofc_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M227_jawsofc_bla/training_LEARNING',
    ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
           ]
    timing_override = [True, True, True]

class OFC_COMPOSITE:
    name = 'OFC_COMPOSITE'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M2_OFC/training', #days 1-4
        'E:/IMPORTANT DATA/DATA_2P/M3_OFC/training', #days 1-4
        'E:/IMPORTANT DATA/DATA_2P/M4_OFC/training', #days 1-6
        'E:/IMPORTANT DATA/DATA_2P/M5_OFC/training', #days 1-4
             ]
    dt_odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    pt_odors = [
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
    timing_override = [True, True, True, True]
    naive_dt_day = [0, 0, 0, 0]
    naive_pt_day = [1, 1, 1, 1]


class MPFC_COMPOSITE:
    name = 'MPFC_COMPOSITE'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M2_MPFC/training',   #days 1-3
        'E:/IMPORTANT DATA/DATA_2P/M3_MPFC/training',   #days 1-3
        'E:/IMPORTANT DATA/DATA_2P/M4_MPFC/training',   #days 1-4
        'E:/IMPORTANT DATA/DATA_2P/M6_MPFC/training'    #days 1-4
    ]
    dt_odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    pt_odors = [
        ['oct'],
        ['oct'],
        ['naive','oct'],
        ['naive','oct'],
    ]
    csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct']
    ]
    timing_override = [True, True, True, True]
    naive_dt_day = [0, 0, 0, 0]
    naive_pt_day = [None, None, 1, 1]

class BLA_JAWS:
    #TODO
    name = 'BLA_JAWS'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M220_jawsbla_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M223_jawsbla_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M224_jawsbla_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M225_jawsbla_ofc/training_LEARNING'
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
    timing_override = [True, True, True, True]

class BLA:
    name = 'BLA'
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
    timing_override = [True, True, True, True]

class BLA_STATE:
    pass

class BLA_CONTEXT:
    pass

class BLA_REVERSAL:
    pass

class BLA_LONGTERM:
    pass



