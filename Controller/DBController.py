from DatabaseAccess import DBConnection as dbCon


# Gets the latest hour from the database Cars table
def getLatestHour():
    return dbCon.getLatestHour()

# Get the lastest Car amount from the database Cars table
def getLatestVehicleAmount():
    return dbCon.getLatestVehicleAmount()

def getLatestWeek():
    return dbCon.getLatestWeek()

# Tells DBConnection to update the latest row in the database Cars table
def updateRow(TableName, Id):
    dbCon.updateRow(TableName, Id)

def getLatestDate():
    return dbCon.getLatestDate()

# Tells DBConnection to insert a new row in the database Cars table
def insertRow(carAmount, check, TableName):
    dbCon.insertRow(carAmount, check, TableName)
