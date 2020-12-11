from DatabaseAccess import DBConnection as dbCon


# Gets the latest hour from the database Cars table
def getLatestHour():
    return dbCon.getLatestHour()


# Get the lastest Car amount from the database Cars table
def getLatestVehicleAmount(vehicleType):
    return dbCon.getLatestVehicleAmount(vehicleType)


def getLatestWeek():
    return dbCon.getLatestWeek()


# Tells DBConnection to update the latest row in the database Cars table
def updateRow(vehicleType):
    dbCon.updateRow(vehicleType)

def IsVehiclesEmpty(vehicleType):
    return dbCon.IsVehiclesEmpty(vehicleType)


def getLatestDate():
    return dbCon.getLatestDate()

def CheckIfTypeExists(typeName, hourStamp, dateStamp):
    return dbCon.CheckIfTypeExists(typeName, hourStamp, dateStamp)


# Tells DBConnection to insert a new row in the database Cars table
def insertRow(carAmount, check, vehicleType):
    dbCon.insertRow(carAmount, check, vehicleType)