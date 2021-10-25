from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.tab import MDTabsBase, MDTabs
from kivymd.uix.toolbar import MDToolbar


class Tab(MDFloatLayout, MDTabsBase):
    def __init__(self, *args, **kwargs):
        super(Tab, self).__init__()
        self.title = kwargs['title']
        self.add_widget(MDLabel(text=kwargs['title'] + ' label', halign="center"))


class Example(MDApp):
    def build(self):
        self.tmp_outer_box = MDBoxLayout(orientation='vertical')
        self.tmp_toolbar = MDToolbar(title="Example Tabs")
        self.tmp_tabs = MDTabs()

        self.tmp_outer_box.add_widget(self.tmp_toolbar)
        self.tmp_outer_box.add_widget(self.tmp_tabs)

        return self.tmp_outer_box

    def on_start(self):
        for i in range(40):
            self.tmp_tabs.add_widget(Tab(title=f"Tab {i}"))


Example().run()
