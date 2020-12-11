from Controller import DBController as dbCtr
import datetime


def getLatestHour():
    return dbCtr.getLatestHour()

def getLatestVehicleAmount(vehicleType):
    return dbCtr.getLatestVehicleAmount(vehicleType)

def updateRow(vehicleType):
    dbCtr.updateRow(vehicleType)

def getLatestDate():
    return dbCtr.getLatestDate()

def insertRow(carAmount, check, vehicleType):
    dbCtr.insertRow(carAmount, check, vehicleType)

def checkDate():
    now = datetime.datetime.now()
    dateStr = now.strftime("%d %b %Y")
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

def CheckIfTypeExists(typeName, hourStamp, dateStamp):
    return dbCtr.CheckIfTypeExists(typeName, hourStamp, dateStamp)

def IsVehiclesEmpty(vehicleType):
    return dbCtr.IsVehiclesEmpty(vehicleType)

# dbInsOrUps står for "database insert or update".
def dbInsOrUpdVehicle(amount, vehicleType):
    if IsVehiclesEmpty(vehicleType) == 0:
        insertRow(amount, True, vehicleType)
    elif CheckIfTypeExists(vehicleType, checkHour(), checkDate()) == 0:
        amount = 1
        insertRow(amount, False, vehicleType)
    else:
        updateRow(vehicleType)
