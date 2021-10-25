"""Kivymd checkbox color change with pytohn only without using kv language.

"""

from kivy.uix.floatlayout import FloatLayout
from kivymd.app import MDApp
from kivymd.uix.selectioncontrol import MDCheckbox
from kivy.metrics import dp


class CustomCheck(MDCheckbox):
    def __init__(self, **kwargs):
        super(CustomCheck, self).__init__(group='group', size_hint=(None, None), size=(dp(48), dp(48)), **kwargs)


class Test(MDApp):

    def __init__(self):
        super(Test, self).__init__()

        self.fl = FloatLayout()

        self.tmp_checkbox = CustomCheck(pos_hint={'center_x': .4, 'center_y': .5}, on_press=self.func1)
        self.tmp_checkbox_2 = CustomCheck(active=True, pos_hint={'center_x': .6, 'center_y': .5}, md_bg_color=(1, 1, 1, 1), on_press=self.func2)

        self.fl.add_widget(self.tmp_checkbox)
        self.fl.add_widget(self.tmp_checkbox_2)

    def func1(self, *args, **kwargs):
        self.tmp_checkbox.update_primary_color(self, (0, 1, 0, 1))

    def func2(self, *args, **kwargs):
        self.tmp_checkbox_2.update_primary_color(self, (0, 0.2, 0.3, 1))

    def build(self):
        return self.fl


Test().run()