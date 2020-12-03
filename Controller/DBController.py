from DatabaseAccess import DBConnection as dbCon


# Gets the latest hour from the database Cars table
def getLatestHour(TableName, Id):
    return dbCon.getLatestHour(TableName, Id)


# Get the lastest Car amount from the database Cars table
def getLatestVehicleAmount(TableName, Id):
    return dbCon.getLatestVehicleAmount(TableName, Id)


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