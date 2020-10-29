from Controller import DBController as dbCtr
import datetime


def getLatestHour():
    return dbCtr.getLatestHour()


def getLatestCarAmount():
    return dbCtr.getLatestCarAmount()


def updateRow():
    dbCtr.updateRow()

def getLatestDate():
    return dbCtr.getLatestDate()


def insertRow(carAmount, check):
    dbCtr.insertRow(carAmount, check)


def checkDate():
    now = datetime.datetime.now()
    dateStr = now.strftime("%d %b %Y ")
    #dateStr = '06 Oct 2020'
    return dateStr


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