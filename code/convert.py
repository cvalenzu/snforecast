from model import SNForecastModel

snmodel = SNForecastModel(units=32, out_steps=3)
snmodel.load_weights("../data")
