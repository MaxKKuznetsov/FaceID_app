from View.view import View

class Controller:
    """
    Класс Controller представляет реализацию контроллера.
    Согласовывает работу представления с моделью.
    """

    def __init__(self, inModel):
        """
        Конструктор принимает ссылку на модель.
        Конструктор создаёт и отображает представление.
        """

        self.state_ID = 0
        self.state_list = ['state1', 'state2']

        # Load data processing class
        self.mModel = inModel

        # Load ui interface class, used to set ui
        self.mView = View(self, self.mModel)

        self.mView.btn_reg.clicked.connect(self.btn_reg_clicked)

        self.mView.show()

    def btn_reg_clicked(self):
        # Use the function of model to process data
        if self.state_ID < len(self.state_list) - 1:
            self.state_ID += 1
        else:
            self.state_ID = 0

        self.mModel.change_state = self.state_list[self.state_ID]


