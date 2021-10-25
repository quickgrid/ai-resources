"""Kivymd python navigation rail without kv language. 

Upon clicking on three dots on toolbar new items are added to the UI. 
This example can be modified to make custom tab switching and more.
"""

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.stacklayout import MDStackLayout
from kivymd.uix.textfield import MDTextField
from kivymd.uix.toolbar import MDToolbar
from kivy.utils import get_color_from_hex


class MyApp(MDApp):
    def __init__(self):
        super(MyApp, self).__init__()
        self.boxlayout1 = MDBoxLayout(orientation="vertical")

        toolbar = MDToolbar(title="DEMO APP")

        toolbar.left_action_items = [["menu", "This is the navigation"]]
        toolbar.right_action_items = [["dots-vertical", lambda x: self.callback_1(x)], ["clock"]]

        self.boxlayout1.add_widget(toolbar)

        self.boxlayout2 = MDBoxLayout()
        self.boxlayout1.add_widget(self.boxlayout2)

        from kivymd.uix.navigationrail import MDNavigationRail, MDNavigationRailItem
        self.navigationrail = MDNavigationRail(
            md_bg_color=get_color_from_hex("#ffffff"),
            # color_normal = get_color_from_hex("#718089"), #crash the app when clicking on an item, don't understand why...
            color_active=get_color_from_hex("#f3ab44"),
            visible="Persistent")

        self.item1 = MDNavigationRailItem(navigation_rail=self.navigationrail, text="Detection", icon='video-stabilization')
        self.item2 = MDNavigationRailItem(navigation_rail=self.navigationrail, text="Dashboard", icon='desktop-mac-dashboard')
        self.item3 = MDNavigationRailItem(navigation_rail=self.navigationrail, text="Settings", icon='cog-outline')
        self.navigationrail.add_widget(self.item1)
        self.navigationrail.add_widget(self.item2)
        self.navigationrail.add_widget(self.item3)

        self.boxlayout2.add_widget(self.navigationrail)

    def callback_1(self, *args, **kwargs):
        sl = MDStackLayout()
        sl.add_widget(MDTextField(hint_text='Enter something'))
        sl.add_widget(MDRaisedButton(text="I AM A BUTTON"))
        self.boxlayout2.add_widget(sl)

    def build(self):
        Clock.schedule_once(self.set_width)
        return self.boxlayout1

    def set_width(self, interval):
        self.navigationrail.size_hint_x = None
        self.navigationrail.width = dp(120)
        #self.navigationrail.md_bg_color = (1, 1, 1, 1)

        self.item1.size_hint_x = None
        self.item1.width = dp(120)
        self.item2.size_hint_x = None
        self.item2.width = dp(120)
        self.item3.size_hint_x = None
        self.item3.width = dp(120)

root = MyApp()
root.run()
