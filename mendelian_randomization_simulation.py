from scenarios.toy_scenarios import HingeLinearScenario, HeaviSideScenario, Zoo, \
    Standardizer, AGMMZoo,MendelianScenario
import numpy as np
np.random.seed(527)

def create_dataset(setting, dir="data/zoo/"):

    # set up model classes, objective, and data scenario
    num_train, num_dev, num_test, n_iv  = setting

    scenario = Standardizer(
        MendelianScenario(n_iv = n_iv))
    scenario.setup(num_train=num_train, num_dev=num_dev, num_test=num_test)
    scenario.info()
    scenario.to_file("data/mendelian/mendelian_{}_0.5_1".format(n_iv))


if __name__ == "__main__":
    for s in [8,16,32]:
        create_dataset([10000,10000,10000,s])
