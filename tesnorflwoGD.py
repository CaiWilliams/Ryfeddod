from Main import Grid, Dispatch
import tensorflow as tf
import pickle








def Setup(NG,Device,lat,lon):
    NG = Grid.Load(NG)
    NG.MatchDates()
    NG.Demand()
    NG.CarbonEmissions()
    NG.PVGISFetch(Device, lat, lon)
    return NG

def ScaleAndRunGD(NG,Solar,SolarBTM,SolarNT=0,SolarBTMNT=0,Target=0):
    NG.Modify('Solar', Scaler=Solar)
    NG.Modify('SolarBTM', Scaler=SolarBTM)
    NG.DynamicScaleingPVGIS('SolarNT', NG.DynamScale, SolarNT)
    NG.DynamicScaleingPVGIS('SolarBTMNT', NG.DynamScale, SolarBTMNT)
    DNG = Dispatch(NG)
    C = (np.sum(DNG.CarbonEmissions) / 2 * (1 * 10 ** -9)) - Target
    return C



NG = Setup('Data/2016RawT.NGM', 'Data/Devices/DSSC.csv', 53.13359, -1.746826)
Target = 50
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
var = tf.Variable(1)
loss = lambda: ScaleAndRunGD(NG, var, var,0,0,Target)
step_count = opt.minimise(loss, [var]).numpy()
print(step_count)