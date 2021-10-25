"""Dropdown menu in KivyMD using python only without using kv language.

"""

from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.screen import MDScreen


class Test(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.screen = MDScreen()

        # self.cust = MDRaisedButton(text="I AM A BUTTON")
        self.cust = MDRaisedButton(text="I AM A BUTTON", on_release=self.custom_func)

        self.menu_items = [
            {
                "text": f"Item {i}",
                "viewclass": "OneLineListItem",
                "on_release": lambda x=f"Item {i}": self.menu_callback(x),
            } for i in range(5)
        ]

        self.menu = MDDropdownMenu(
            caller=self.cust,
            items=self.menu_items,
            width_mult=4,
        )

        # self.cust.on_release=self.custom_func
        self.screen.add_widget(self.cust)


    def custom_func(self, *args, **kwargs):
        self.menu.open()

    def menu_callback(self, text_item):
        print(text_item)

    def build(self):
        return self.screen


Test().run()
