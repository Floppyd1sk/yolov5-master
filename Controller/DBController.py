from DatabaseAccess import DBConnection as dbCon


# Gets the latest hour from the database Cars table
def getLatestHour():
    return dbCon.getLatestHour()


# Get the lastest Car amount from the database Cars table
def getLatestCarAmount():
    return dbCon.getLatestCarAmount()

def getLatestWeek():
    return dbCon.getLatestWeek()

# Tells DBConnection to update the latest row in the database Cars table
def updateRow():
    dbCon.updateRow()

def getLatestDate():
    return dbCon.getLatestDate()

# Tells DBConnection to insert a new row in the database Cars table
def insertRow(carAmount, check):
    dbCon.insertRow(carAmount, check)
