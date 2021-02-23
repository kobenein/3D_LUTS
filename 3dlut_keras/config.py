import json

def getConf(path):
    with open(path) as config_file:
        CONF = json.load(config_file)
    
    assert len(CONF["name"]) > 0, 'plz set experiment name'

    if isinstance(CONF["parameter"]["Optimizers"],list):
        CONF["parameter"]["Optimizers"] = CONF["parameter"]["Optimizers"][0]

    if isinstance(CONF["parameter"]["schedules"],list):
        CONF["parameter"]["schedules"] = CONF["parameter"]["schedules"][0]

    # for k,v in CONF.items():
    #     if isinstance(v,dict):
    #         print(f"[{k}]"}
    #         for kk,vv in v.items():
    #             print(f"    {kk}:{vv}"}
    #     else:
    #         print(f"{k}:{v}"}

    return CONF

