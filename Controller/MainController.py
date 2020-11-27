from Controller import DBController as dbCtr
import datetime


def getLatestHour():
    return dbCtr.getLatestHour()


def getLatestCarAmount():
    return dbCtr.getLatestCarAmount()

def getLatestTruckAmount():
    return dbCtr.getLatestTruckAmount()


def updateRow(TableName):
    dbCtr.updateRow(TableName)

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
def dbInsOrUpd(carAmount):
    if getLatestHour() is None or getLatestDate() is None:
        insertRow(carAmount, True)
    elif not getLatestHour() == checkHour() and not getLatestDate() == checkDate():
        carAmount = 0
        insertRow(carAmount, False)
    elif not getLatestHour() == checkHour() and getLatestDate() == checkDate():
        carAmount = 0
        insertRow(carAmount, False)
    elif getLatestHour() == checkHour() and getLatestDate() == checkDate():
        updateRow()

def dbInsOrUpdCar(carAmount):
    if getLatestHour() is None or getLatestDate() is None:
        insertRow(carAmount, True, "Cars")
    elif not getLatestHour() == checkHour() and not getLatestDate() == checkDate():
        carAmount = 0
        insertRow(carAmount, False, "Cars")
    elif not getLatestHour() == checkHour() and getLatestDate() == checkDate():
        carAmount = 0
        insertRow(carAmount, False, "Cars")
    elif getLatestHour() == checkHour() and getLatestDate() == checkDate():
        updateRow("Cars")

def dbInsOrUpdTruck(carAmount):
    if getLatestHour() is None or getLatestDate() is None:
        insertRow(carAmount, True, "Trucks")
    elif not getLatestHour() == checkHour() and not getLatestDate() == checkDate():
        carAmount = 0
        insertRow(carAmount, False, "Trucks")
    elif not getLatestHour() == checkHour() and getLatestDate() == checkDate():
        carAmount = 0
        insertRow(carAmount, False, "Trucks")
    elif getLatestHour() == checkHour() and getLatestDate() == checkDate():
        updateRow("Trucks")
