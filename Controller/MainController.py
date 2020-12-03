from Controller import DBController as dbCtr
import datetime


def getLatestHour(TableName, Id):
    return dbCtr.getLatestHour(TableName, Id)

def getLatestVehicleAmount(TableName, Id):
    return dbCtr.getLatestVehicleAmount(TableName, Id)

def updateRow(TableName, Id):
    dbCtr.updateRow(TableName, Id)

def getLatestDate():
    return dbCtr.getLatestDate()

def insertRow(carAmount, check, TableName):
    dbCtr.insertRow(carAmount, check, TableName)

def checkDate():
    now = datetime.datetime.now()
    dateStr = now.strftime("%d %b %Y ")
    #dateStr = '06 Oct 2020'
    return dateStr

def checkWeek():
    now = datetime.date.now()
    year,week_num,day_of_week = now.isocalendar()
    return week_num

def checkHour():
    now = datetime.datetime.now()
    hourStr = now.strftime("%H")
    return hourStr

# dbInsOrUps står for "database insert or update".
def dbInsOrUpdCar(amount):
    tableName = 'Cars'
    id = 'CarId'
    if getLatestHour(tableName, id) is None or getLatestDate() is None:
        insertRow(amount, True, tableName)
    elif not getLatestHour(tableName, id) == checkHour():
        if getLatestDate() == checkDate() or not getLatestDate() == checkDate():
            amount = 1
            insertRow(amount, False, tableName)
    elif getLatestHour(tableName, id) == checkHour() and getLatestDate() == checkDate():
        updateRow(tableName, id)

def dbInsOrUpdTruck(amount):
    tableName = 'Trucks'
    id = 'TruckId'
    if getLatestHour(tableName, id) is None or getLatestDate() is None:
        insertRow(amount, True, tableName)
    elif not getLatestHour(tableName, id) == checkHour():
        if getLatestDate() == checkDate() or not getLatestDate() == checkDate():
            amount = 1
            insertRow(amount, False, tableName)
    elif getLatestHour(tableName, id) == checkHour() and getLatestDate() == checkDate():
        updateRow(tableName, id)

def dbInsOrUpdMotorcycle(amount):
    tableName = 'Motorcycles'
    id = 'MotorcycleId'
    if getLatestHour(tableName, id) is None or getLatestDate() is None:
        insertRow(amount, True, tableName)
    elif not getLatestHour(tableName, id) == checkHour():
        if getLatestDate() == checkDate() or not getLatestDate() == checkDate():
            amount = 1
            insertRow(amount, False, tableName)
    elif getLatestHour(tableName, id) == checkHour() and getLatestDate() == checkDate():
        updateRow(tableName, id)

