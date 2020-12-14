from Controller import DBController as dbCtr
import datetime

def getLatestVehicleAmount(vehicleType):
    return dbCtr.getLatestVehicleAmount(vehicleType)

def insertRow(vehicleAmount, check, vehicleType):
    dbCtr.insertRow(vehicleAmount, check, vehicleType)

# dbInsOrUps står for "database insert or update".
def dbInsOrUpdVehicle(vehicleType):
        insertRow(1, False, vehicleType)
