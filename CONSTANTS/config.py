from tools import file_io


class Config:
    save_mat_f = file_io.save_text
    load_mat_f = file_io.load_text

    save_cons_f = file_io.save_pickle
    load_cons_f = file_io.load_pickle

