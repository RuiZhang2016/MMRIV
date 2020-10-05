from scenarios.toy_scenarios import Standardizer, MendelianScenario
import numpy as np
np.random.seed(527)

def create_dataset(setting, dir="data/zoo/"):

    # set up model classes, objective, and data scenario
    num_train, num_dev, num_test, n_iv= setting

    scenario = Standardizer(
        MendelianScenario(n_iv = n_iv))
    scenario.setup(num_train=num_train, num_dev=num_dev, num_test=num_test)
    scenario.info()
    # scenario.to_file("data/mendelian/mendelian_{}_{}_{}".format(n_iv,c2,c1)) # #IV - c2 - c1


if __name__ == "__main__":
    # for s in [8,16,32]:
    #     create_dataset([10000,10000,10000,s,1,1])
    #
    for s in [8,16,32]:
        create_dataset([10000,10000,10000,s])
    #
    # for s in [0.5,1,2]:
    #     create_dataset([10000,10000,10000,16,1,s])
