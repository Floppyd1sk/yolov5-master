from DatabaseAccess import DBConnection as dbCon

# Get the lastest Car amount from the database Cars table
def getLatestVehicleAmount(vehicleType):
    return dbCon.getLatestVehicleAmount(vehicleType)

# Tells DBConnection to insert a new row in the database Cars table
def insertRow(vehicleAmount, check, vehicleType):
    dbCon.insertRow(vehicleAmount, check, vehicleType)