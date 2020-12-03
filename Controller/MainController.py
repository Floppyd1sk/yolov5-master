from Controller import DBController as dbCtr
import datetime


def getLatestHour():
    return dbCtr.getLatestHour()

def getLatestVehicleAmount():
    return dbCtr.getLatestVehicleAmount()

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
def dbInsOrUpdCar(carAmount):
    if getLatestHour() is None or getLatestDate() is None:
        insertRow(carAmount, True, "Cars")
    elif not getLatestHour() == checkHour():
        if getLatestDate() == checkDate() or not getLatestDate() == checkDate():
            carAmount = 0
            insertRow(carAmount, False, "Cars")
    elif getLatestHour() == checkHour() and getLatestDate() == checkDate():
        updateRow("Cars", "CarId")

def dbInsOrUpdTruck(carAmount):
    if getLatestHour() is None or getLatestDate() is None:
        insertRow(carAmount, True, "Trucks")
    elif not getLatestHour() == checkHour():
        if getLatestDate() == checkDate() or not getLatestDate() == checkDate():
            carAmount = 0
            insertRow(carAmount, False, "Trucks")
    elif getLatestHour() == checkHour() and getLatestDate() == checkDate():
        updateRow("Trucks", "TruckId")
