
from collections import defaultdict
import numpy as np
import filter
import reduce

def get_valence_sig(res):
    #TODO: not tested yet
    def _helper(res):
        assert res['odor_valence'][0] == 'CS+', 'wrong odor'
        assert res['odor_valence'][1] == 'CS-', 'wrong odor'
        on = res['DAQ_O_ON_F'][0]
        off = res['DAQ_W_ON_F'][0]
        sig_p = res['sig'][0]
        sig_m = res['sig'][1]
        dff_p = res['dff'][0]
        dff_m = res['dff'][1]
        sig_p_mask = sig_p == 1
        sig_m_mask = sig_m == 1
        dff_mask = dff_p - dff_m
        dff_mask = np.mean(dff_mask[:, on:off], axis=1)
        p = [a and b for a, b in zip(sig_p_mask, dff_mask>0)]
        m = [a and b for a, b in zip(sig_m_mask, dff_mask<0)]
        return np.array(p), np.array(m)

    mice = np.unique(res['mouse'])
    res = filter.filter(res, filter_dict={'odor_valence':['CS+','CS-']})
    sig_res = reduce.new_filter_reduce(res, reduce_key='sig', filter_keys=['mouse','day','odor_valence'])
    dff_res = reduce.new_filter_reduce(res, reduce_key='dff', filter_keys=['mouse','day','odor_valence'])
    sig_res['dff'] = dff_res['dff']

    reversal_res = defaultdict(list)
    day_strs = ['Lrn', 'Rev']
    for mouse in mice:
        mouse_res = filter.filter(sig_res, filter_dict={'mouse': mouse})
        days = np.unique(mouse_res['day'])
        p_list = []
        m_list = []
        for i, day in enumerate(days):
            mouse_day_res = filter.filter(mouse_res, filter_dict={'day': day})
            p, m = _helper(mouse_day_res)
            reversal_res['mouse'].append(mouse)
            reversal_res['mouse'].append(mouse)
            reversal_res['day'].append(day_strs[i])
            reversal_res['day'].append(day_strs[i])
            reversal_res['odor_valence'].append('CS+')
            reversal_res['odor_valence'].append('CS-')
            reversal_res['sig'].append(p)
            reversal_res['sig'].append(m)
            reversal_res['Fraction'].append(np.mean(p))
            reversal_res['Fraction'].append(np.mean(m))
            p_list.append(p)
            m_list.append(m)
    for key, val in reversal_res.items():
        reversal_res[key] = np.array(val)