# -*- coding: UTF-8 -*-


def list_to_dict(even_list):
    ret = {}
    if even_list:
        for i in range(0, len(even_list), 2):
            try:
                ret[even_list[i]] = int(even_list[i + 1])
                continue
            except ValueError:
                pass
            try:
                ret[even_list[i]] = float(even_list[i + 1])
                continue
            except ValueError:
                pass
            if even_list[i + 1].lower() == 'true':
                ret[even_list[i]] = True
            elif even_list[i + 1].lower() == 'false':
                ret[even_list[i]] = False
            else:
                ret[even_list[i]] = even_list[i + 1]
    return ret


def dict_to_oneline(to_oneline: dict) -> str:
    return ', '.join([
        '%s=%s' % (key, value) for key, value in to_oneline.items()])
