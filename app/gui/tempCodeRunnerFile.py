    """Jump calendar to today's date"""
        today = date.today()
        qdate = QDate(today.year, today.month, today.day)
        self.calendar.setSelectedDate(qdate)
        self.selected_date = today
        self.load_existing_data()

