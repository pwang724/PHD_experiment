from collections import namedtuple

#CSP are always the first two odors

def all_conditions():
    out = [
        PIR, PIR_NAIVE,
        OFC, OFC_COMPOSITE, OFC_LONGTERM, OFC_STATE, OFC_CONTEXT, OFC_REVERSAL,
        BLA, BLA_LONGTERM, BLA_STATE, BLA_CONTEXT, BLA_REVERSAL,
        BLA_JAWS,
        MPFC_COMPOSITE
    ]
    return out

class PIR:
    name = 'PIR'
    paths = ['I:/IMPORTANT DATA/DATA_2P/M183/training_LEARNING',
             'I:/IMPORTANT DATA/DATA_2P/M184/training_LEARNING',
             'I:/IMPORTANT DATA/DATA_2P/M199_pir/training_LEARNING',
             'I:/IMPORTANT DATA/DATA_2P/M200_pir/training_LEARNING',
             'I:/IMPORTANT DATA/DATA_2P/M201_pir/training_LEARNING',
             'I:/IMPORTANT DATA/DATA_2P/M202_pir/training_LEARNING']
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
    training_start_day = [0, 0, 0, 0, 0, 0]
    min_neurons = 46

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
    csp = [
        ['2pe', 'ben', 'msy', 'oct'],
        ['2pe', 'ben', 'msy', 'oct'],
        ['2pe', 'ben', 'msy', 'oct'],
        ['2pe', 'ben', 'msy', 'oct']
           ]
    timing_override = [True, True, True, True]
    training_start_day = [0, 0, 0, 0]
    min_neurons = 46

class PIR_CONTEXT:
    name = 'PIR_CONTEXT'
    paths = [
        r'I:\IMPORTANT DATA\DATA_2P\M183\training_CONTEXT',
        r'I:\IMPORTANT DATA\DATA_2P\M184\training_CONTEXT',
             ]
    odors = [
        ['pin', 'msy', '4mt', 'lim'],
        ['pin', 'msy', '4mt', 'lim'],
    ]
    csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
           ]
    timing_override = [True, True]
    training_start_day = [0, 0]
    # min_neurons = 46


class OFC:
    name = 'OFC'
    paths = [
        'I:/IMPORTANT DATA/DATA_2P/M187_ofc/training_LEARNING',
        'I:/IMPORTANT DATA/DATA_2P/M188_ofc/training_LEARNING',
        'I:/IMPORTANT DATA/DATA_2P/M206_ofc/training_LEARNING',
        'I:/IMPORTANT DATA/DATA_2P/M233_ofc/training_LEARNING',
        'I:/IMPORTANT DATA/DATA_2P/M234_ofc/training_LEARNING'
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
    training_start_day = [1, 1, 1, 1, 1]
    min_neurons= 52

class OFC_STATE:
    name = 'OFC_STATE'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M187_ofc/training_STATE',
        'E:/IMPORTANT DATA/DATA_2P/M188_ofc/training_STATE',
        'E:/IMPORTANT DATA/DATA_2P/M206_ofc/training_STATE',
        'E:/IMPORTANT DATA/DATA_2P/M233_ofc/training_STATE',
        'E:/IMPORTANT DATA/DATA_2P/M234_ofc/training_STATE'
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
        ['pin', 'msy']
           ]
    timing_override = [True, True, True, True, True]
    training_start_day = [0, 0, 0, 0, 0]

class OFC_CONTEXT:
    name = 'OFC_CONTEXT'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M187_ofc/training_CONTEXT',
        'E:/IMPORTANT DATA/DATA_2P/M188_ofc/training_CONTEXT',
        'E:/IMPORTANT DATA/DATA_2P/M206_ofc/training_CONTEXT',
        'E:/IMPORTANT DATA/DATA_2P/M233_ofc/training_CONTEXT'
    ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['euy', 'lim'],
        ['euy', 'lim'],
        ['euy', 'lim'],
        ['euy', 'lim']
           ]
    timing_override = [True, True, True, True]
    training_start_day = [0, 0, 0, 0]

class OFC_REVERSAL:
    name = 'OFC_REVERSAL'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M187_ofc/training_REVERSAL',
        'E:/IMPORTANT DATA/DATA_2P/M188_ofc/training_REVERSAL',
        'E:/IMPORTANT DATA/DATA_2P/M206_ofc/training_REVERSAL',
        'E:/IMPORTANT DATA/DATA_2P/M233_ofc/training_REVERSAL',
        'E:/IMPORTANT DATA/DATA_2P/M234_ofc/training_REVERSAL'
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
        ['pin', 'msy']
           ]
    timing_override = [True, True, True, True, True]
    training_start_day = [0, 0, 0, 0, 0]

class OFC_LONGTERM:
    name = 'OFC_LONGTERM'
    paths = [
        'I:/IMPORTANT DATA/DATA_2P/M230_ofc/training_LONGTERM',
        'I:/IMPORTANT DATA/DATA_2P/M232_ofc/training_LONGTERM',
        'I:/IMPORTANT DATA/DATA_2P/M239_ofc/training_LONGTERM',
        'I:/IMPORTANT DATA/DATA_2P/M241_ofc/training_LONGTERM'
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
    training_start_day = [0, 0, 0, 0]
    min_neurons = 58

class OFC_COMPOSITE:
    name = 'OFC_COMPOSITE'
    paths = [
        'I:/IMPORTANT DATA/DATA_2P/M2_OFC/training', #days 1-4
        'I:/IMPORTANT DATA/DATA_2P/M3_OFC/training', #days 1-4
        'I:/IMPORTANT DATA/DATA_2P/M4_OFC/training', #days 1-6
        'I:/IMPORTANT DATA/DATA_2P/M5_OFC/training', #days 1-4
             ]
    dt_odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    dt_csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
    ]
    pt_odors = [
        ['naive', 'oct'],
        ['naive', 'oct'],
        ['naive', 'oct'],
        ['naive', 'oct']
    ]
    pt_csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct']
    ]
    timing_override = [True, True, True, True]
    training_start_day = [1, 1, 1, 1]
    naive_dt_day = [0, 0, 0, 0]
    naive_pt_day = [1, 1, 1, 1]
    last_pt_day = [3, 3, 5, 3]
    first_dt_day = [4, 4, 6, 4]


class MPFC_COMPOSITE:
    name = 'MPFC_COMPOSITE'
    paths = [
        'I:/IMPORTANT DATA/DATA_2P/M2_MPFC/training',   #days 1-3
        'I:/IMPORTANT DATA/DATA_2P/M3_MPFC/training',   #days 1-3
        'I:/IMPORTANT DATA/DATA_2P/M4_MPFC/training',   #days 1-4
        'I:/IMPORTANT DATA/DATA_2P/M6_MPFC/training'    #days 1-4
    ]
    dt_odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]

    dt_csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
    ]

    pt_odors = [
        ['naive','oct'],
        ['naive','oct'],
        ['naive','oct'],
        ['naive','oct'],
    ]
    pt_csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct']
    ]
    timing_override = [True, True, True, True]
    training_start_day = [1, 1, 1, 1]
    naive_dt_day = [0, 0, 0, 0]
    naive_pt_day = [1, 1, 1, 1]
    last_pt_day = [2, 2, 3, 3]
    first_dt_day = [3, 3, 4, 4]
    min_neurons = 49

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
    training_start_day = [0, 0, 0, 0]

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
    training_start_day = [1, 1, 1, 1]
    min_neurons = 22

class BLA_STATE:
    name = 'BLA_STATE'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M211_bla/training_STATE',
        'E:/IMPORTANT DATA/DATA_2P/M212_bla/training_STATE'
        ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['euy', 'lim'],
        ['euy', 'lim']
           ]
    timing_override = [True, True]

class BLA_CONTEXT:
    name = 'BLA_CONTEXT'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M211_bla/training_CONTEXT',
        'E:/IMPORTANT DATA/DATA_2P/M212_bla/training_CONTEXT'
        ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['pin','msy'],
        ['pin','msy']
           ]
    timing_override = [True, True]

class BLA_REVERSAL:
    name = 'BLA_REVERSAL'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M211_bla/training_REVERSAL',
        'E:/IMPORTANT DATA/DATA_2P/M212_bla/training_REVERSAL'
        ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['pin','msy'],
        ['pin','msy']
           ]
    timing_override = [True, True]
class BLA_LONGTERM:
    name = 'BLA_LONGTERM'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M211_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M212_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M213_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M214_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M260_ofc_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M261_bla/training_LEARNING'
        ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
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
           ['pin', 'msy']
           ]
    timing_override = [True, True, True, True, True, True]
    training_start_day = [1, 1, 1, 1, 1, 1]

class BEHAVIOR_OFC_MUSH_YFP:
    name = 'BEHAVIOR_OFC_YFP_MUSH'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\SINGLE_PHASE_OFC_YFP\PN06',
        'I:\IMPORTANT DATA\DATA_2P\SINGLE_PHASE_OFC_YFP\PN08',
        'I:\IMPORTANT DATA\DATA_2P\SINGLE_PHASE_OFC_YFP\PN09',
        'I:\IMPORTANT DATA\DATA_2P\SINGLE_PHASE_OFC_YFP\PN10',
        'I:\IMPORTANT DATA\DATA_2P\SINGLE_PHASE_OFC_YFP\M400_MUSH_YFP',
        'I:\IMPORTANT DATA\DATA_2P\SINGLE_PHASE_OFC_YFP\M402_MUSH_YFP',
        'I:\IMPORTANT DATA\DATA_2P\SINGLE_PHASE_OFC_YFP\M418_OFC_MUSH_YFP'
    ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
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
        ['pin', 'msy'],
        ['pin', 'msy'],
           ]
    timing_override = [True, True, True, True, True, True, True]
    training_start_day = [0, 0, 0, 0, 0, 0, 0]

class BEHAVIOR_OFC_MUSH_JAWS_UNUSED:
    name = 'BEHAVIOR_OFC_JAWS_MUSH_UNUSED'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\PN04',
        'I:\IMPORTANT DATA\DATA_2P\PN05',
        'I:\IMPORTANT DATA\DATA_2P\PN11',
        'I:\IMPORTANT DATA\DATA_2P\PN12',
        'I:\IMPORTANT DATA\DATA_2P\PN13',
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
    timing_override = [True] * 5
    training_start_day = [0, 0, 0, 0, 0]

class BEHAVIOR_OFC_MUSH_JAWS:
    name = 'BEHAVIOR_OFC_JAWS_MUSH'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P/M222_jawsofc_bla/training_LEARNING',
        'I:\IMPORTANT DATA\DATA_2P/M226_jawsofc_bla/training_LEARNING',
        'I:\IMPORTANT DATA\DATA_2P/M227_jawsofc_bla/training_LEARNING',
        'I:\IMPORTANT DATA\DATA_2P\SINGLE_PHASE_OFC_JAWS_BLA_IMAGING/M221_jawsofc_bla',
    ]
    odors = [
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
    ]
    timing_override = [True] * 4
    training_start_day = [0] * 4

class BEHAVIOR_OFC_MUSH_HALO:
    name = 'BEHAVIOR_OFC_HALO_MUSH'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\SINGLE_PHASE_OFC_HALO\PN14',
        'I:\IMPORTANT DATA\DATA_2P\SINGLE_PHASE_OFC_HALO\PN15',
        'I:\IMPORTANT DATA\DATA_2P\SINGLE_PHASE_OFC_HALO\PN16',
        'I:\IMPORTANT DATA\DATA_2P\SINGLE_PHASE_OFC_HALO\PN17',
        'I:\IMPORTANT DATA\DATA_2P\SINGLE_PHASE_OFC_HALO\PN18',
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
    timing_override = [True] * 5
    training_start_day = [0] * 5


class BEHAVIOR_OFC_YFP_PRETRAINING:
    name = 'BEHAVIOR_OFC_YFP_PRETRAINING'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_YFP/C1_M247',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_YFP/C2_M248',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_YFP/C3_M249',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_YFP/C4_M250',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_YFP\M403_PT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_YFP\M404_PT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_YFP\M405_PT_YFP'
    ]
    dt_odors = [
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
    ]
    dt_csp = [
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
    ]
    pt_odors = [
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    pt_csp = [
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    timing_override = [True, True, True, True, True, True, True] #TODO: CHECK THIS IS TRUE
    training_start_day = [0, 0, 0, 0, 0, 0, 0]
    last_pt_day = [2, 2, 2, 2, 2, 2, 2]

class BEHAVIOR_OFC_JAWS_PRETRAINING:
    name = 'BEHAVIOR_OFC_JAWS_PRETRAINING'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_JAWS\J3_M245',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_JAWS\J5_M251',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_JAWS\J6_M252',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_JAWS\J7_M253',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_JAWS\J8_M262',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_JAWS\J9_M263',
    ]
    dt_odors = [
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
    ]
    dt_csp = [
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
    ]
    pt_odors = [
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
    ]
    pt_csp = [
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
    ]
    timing_override = [True, True, True, True, True, True]
    training_start_day = [0, 0, 0, 0, 0, 0]
    last_pt_day = [4, 3, 3, 1, 2, 2]

class BEHAVIOR_OFC_HALO_PRETRAINING:
    name = 'BEHAVIOR_OFC_HALO_PRETRAINING'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_HALO\M412_PT_HALO',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_HALO\M414_PT_HALO',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_HALO\M416_OFC_PT_HALO',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_OFC_HALO\M417_OFC_PT_HALO',
    ]
    dt_odors = [
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
    ]
    dt_csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
    ]
    pt_odors = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    pt_csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    timing_override = [True, True, True, True]
    training_start_day = [0, 0, 0, 0]
    # last_pt_day = [4, 3, 3, 1, 2, 2]

class BEHAVIOR_OFC_YFP_DISCRIMINATION:
    name = 'BEHAVIOR_OFC_YFP_DISCRIMINATION'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_OFC_YFP\M406_PTDT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_OFC_YFP\M408_PTDT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_OFC_YFP\M410_PTDT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_OFC_YFP\M411_PTDT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_OFC_YFP\M413_PTDT_YFP'
    ]
    dt_odors = [
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
    ]
    dt_csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
    ]
    pt_odors = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    pt_csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    timing_override = [True, True, True, True, True] #TODO: CHECK THIS IS TRUE
    training_start_day = [0, 0, 0, 0, 0]
    # last_pt_day = [2, 2, 2, 2, 2, 2, 2]

class BEHAVIOR_OFC_JAWS_DISCRIMINATION:
    name = 'BEHAVIOR_OFC_JAWS_DISCRIMINATION'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\JD8_254',
        'I:\IMPORTANT DATA\DATA_2P\JD9_255',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_OFC_JAWS/JD10_256',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_OFC_JAWS/JD11_257',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_OFC_JAWS/JD12_258',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_OFC_JAWS/JD13_259',
    ]
    dt_odors = [
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
    ]
    dt_csp = [
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
    ]
    pt_odors = [
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
    ]
    pt_csp = [
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
    ]
    timing_override = [True, True, True, True,True,True]
    training_start_day = [0, 0, 0,0,0, 0]
    # last_pt_day = [3, 3, 2, 2]

class BEHAVIOR_OFC_OUTPUT_CHANNEL:
    name = 'BEHAVIOR_OFC_OUTPUT_CHANNEL'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\OUTPUT_CHANNEL/AH08_OUTPUT_CHANNEL',
        'I:\IMPORTANT DATA\DATA_2P\OUTPUT_CHANNEL/AH10_OUTPUT_CHANNEL',
        'I:\IMPORTANT DATA\DATA_2P\OUTPUT_CHANNEL/AH11_OUTPUT_CHANNEL',
        'I:\IMPORTANT DATA\DATA_2P\OUTPUT_CHANNEL/AH14_OUTPUT_CHANNEL',
        'I:\IMPORTANT DATA\DATA_2P\OUTPUT_CHANNEL/AH19_OUTPUT_CHANNEL',
        'I:\IMPORTANT DATA\DATA_2P\OUTPUT_CHANNEL/AH20_OUTPUT_CHANNEL',
    ]
    pt_odors = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    pt_csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    dt_odors = [
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    dt_csp = [
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    timing_override = [True] * 6
    training_start_day = [0] * 6

class BEHAVIOR_OFC_OUTPUT_YFP:
    name = 'BEHAVIOR_OFC_OUTPUT_YFP'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\OUTPUT_YFP/AH16_OUTPUT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\OUTPUT_YFP/AH18_OUTPUT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\OUTPUT_YFP/AH21_OUTPUT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\OUTPUT_YFP/AH24_OUTPUT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\OUTPUT_YFP/AH26_OUTPUT_YFP',
    ]
    pt_odors = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    pt_csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    dt_odors = [
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    dt_csp = [
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    timing_override = [True] * 5
    training_start_day = [0] * 5

class BEHAVIOR_MPFC_HALO_DISCRIMINATION:
    name = 'BEHAVIOR_MPFC_HALO_DISCRIMINATION'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_MPFC_HALO\AH45_MPFC_DT_HALO',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_MPFC_HALO\AH46_MPFC_DT_HALO',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_MPFC_HALO\AH51_MPFC_DT_HALO',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_MPFC_HALO\AH52_MPFC_DT_HALO',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_MPFC_HALO\AH53_MPFC_DT_HALO',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_MPFC_HALO\AH54_MPFC_DT_HALO',
    ]
    pt_odors = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    pt_csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    dt_odors = [
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
    ]
    dt_csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
    ]
    timing_override = [True, True, True, True, True, True]
    training_start_day = [0, 0, 0, 0, 0, 0]

class BEHAVIOR_MPFC_YFP_DISCRIMINATION:
    name = 'BEHAVIOR_MPFC_YFP_DISCRIMINATION'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_MPFC_YFP\AH58_MPFC_DT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_MPFC_YFP\AH59_MPFC_DT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_MPFC_YFP\AH60_MPFC_DT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_MPFC_YFP\AH62_MPFC_DT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_DISCRIMINATION_MPFC_YFP\AH64_MPFC_DT_YFP',
    ]
    pt_odors = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    pt_csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    dt_odors = [
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
        ['msy', 'pin', 'euy', 'lim'],
    ]
    dt_csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
    ]
    timing_override = [True, True, True, True, True]
    training_start_day = [0, 0, 0, 0, 0]

class BEHAVIOR_MPFC_HALO_PRETRAINING:
    name = 'BEHAVIOR_MPFC_HALO_PRETRAINING'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_MPFC_HALO\AH42_MPFC_PT_HALO',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_MPFC_HALO\AH43_MPFC_PT_HALO',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_MPFC_HALO\AH44_MPFC_PT_HALO',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_MPFC_HALO\AH50_MPFC_PT_HALO',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_MPFC_HALO\AH55_MPFC_PT_HALO',
    ]
    pt_odors = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    pt_csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    dt_odors = [
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    dt_csp = [
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    timing_override = [True] * 5
    training_start_day = [0] * 5

class BEHAVIOR_MPFC_YFP_PRETRAINING:
    name = 'BEHAVIOR_MPFC_YFP_PRETRAINING'
    paths = [
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_MPFC_YFP\AH57_MPFC_PT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_MPFC_YFP\AH65_MPFC_PT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_MPFC_YFP\AH66_MPFC_PT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_MPFC_YFP\AH67_MPFC_PT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_MPFC_YFP\AH68_MPFC_PT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_MPFC_YFP\AH69_MPFC_PT_YFP',
        'I:\IMPORTANT DATA\DATA_2P\TWO_PHASE_PRETRAINING_MPFC_YFP\AH70_MPFC_PT_YFP',
    ]
    pt_odors = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    pt_csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct'],
    ]
    dt_odors = [
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    dt_csp = [
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    timing_override = [True] * 7
    training_start_day = [0] * 7



