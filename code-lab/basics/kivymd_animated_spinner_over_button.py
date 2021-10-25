"""Kivymd python example of showing an animated spinner over button with click event.

When button is clicked if there is spinner it is removed and on another click
if there is no spinner attached then it is reattached. This code uses python without
using kv language.
"""


from kivy.metrics import dp
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton, MDFillRoundFlatIconButton
from kivymd.uix.screen import MDScreen
from kivymd.uix.spinner import MDSpinner


class Test(MDApp):
    def __init__(self):
        super(Test, self).__init__()

        self.click_tracker = 0

        self.main_screen = MDScreen()

        self.cust_button = MDFillRoundFlatIconButton(
            text="MDRAISEDBUTTON",
            md_bg_color=(1, 0, 1, 1),
            #size_hint=(None, None),
            #size=(300, 300),
            pos_hint={'center_x':0.5, 'center_y':0.5},
            font_size=dp(16),
            on_press=self.cust_button_func
        )


        self.cust_spinner = MDSpinner(
            size_hint=(None, None),
            size=(dp(28), dp(28)),
            #pos_hint={'center_x':0.1, 'center_y':0.1},
            active=True,
            line_width=3,
            palette=[
                [0.28627450980392155, 0.8431372549019608, 0.596078431372549, 1],
                [0.3568627450980392, 0.3215686274509804, 0.8666666666666667, 1],
                [0.8862745098039215, 0.36470588235294116, 0.592156862745098, 1],
                [0.8784313725490196, 0.9058823529411765, 0.40784313725490196, 1],
            ]
        )

        self.cust_button.add_widget(self.cust_spinner)

        self.main_screen.add_widget(self.cust_button)

    def cust_button_func(self, *args, **kwargs):
        if self.click_tracker == 0:
            self.cust_button.remove_widget(self.cust_spinner)
            self.click_tracker = not self.click_tracker
        else:
            self.cust_button.add_widget(self.cust_spinner)
            self.click_tracker = not self.click_tracker

    def build(self):
        return self.main_screen


Test().run()