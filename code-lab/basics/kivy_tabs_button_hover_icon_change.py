"""Kivy multiple tabs and button on hover cursor change with python only.

This example does not use kv language. This is only for those who do not know or use kivymd.
There is hovering problem when sometimes it does not change to arrow.

References:
    https://gist.github.com/opqopq/15c707dc4cffc2b6455f
"""
from kivy.app import App
from kivy.properties import BooleanProperty, ObjectProperty
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem


class HoverBehavior(object):
    """Hover behavior.
    :Events:
        `on_enter`
            Fired when mouse enter the bbox of the widget.
        `on_leave`
            Fired when the mouse exit the widget
    """

    hovered = BooleanProperty(False)
    border_point= ObjectProperty(None)
    '''Contains the last relevant point received by the Hoverable. This can
    be used in `on_enter` or `on_leave` in order to know where was dispatched the event.
    '''

    def __init__(self, **kwargs):
        self.register_event_type('on_enter')
        self.register_event_type('on_leave')
        Window.bind(mouse_pos=self.on_mouse_pos)
        super(HoverBehavior, self).__init__(**kwargs)

    def on_mouse_pos(self, *args):
        if not self.get_root_window():
            return # do proceed if I'm not displayed <=> If have no parent
        pos = args[1]
        #Next line to_widget allow to compensate for relative layout
        inside = self.collide_point(*self.to_widget(*pos))
        if self.hovered == inside:
            #We have already done what was needed
            return
        self.border_point = pos
        self.hovered = inside
        if inside:
            self.dispatch('on_enter')
        else:
            self.dispatch('on_leave')

    def on_enter(self):
        pass

    def on_leave(self):
        pass


from kivy.factory import Factory
Factory.register('HoverBehavior', HoverBehavior)


class HoverButton(Button, HoverBehavior):
    def on_enter(self, *args):
        print(self.text)
        print("You are in, through this point", self.border_point)
        Window.set_system_cursor('hand')

    def on_leave(self, *args):
        print("You left through this point", self.border_point)
        Window.set_system_cursor('arrow')


class HoverTabs(TabbedPanelItem, HoverBehavior):
    def on_enter(self, *args):
        print(self.text)
        print("You are in, through this point", self.border_point)
        Window.set_system_cursor('hand')

    def on_leave(self, *args):
        print("You left through this point", self.border_point)
        Window.set_system_cursor('arrow')


class ButtonApp(App):
    def __init__(self, **kwargs):
        super(ButtonApp, self).__init__(**kwargs)


        tmp_btn = HoverButton(text='BUTTON 1', size_hint=(None, None), size=(200, 60), color=(0, 1, 0, 1), font_size=dp(20),
                              on_press=self.do_sum)
        tmp_btn2 = HoverButton(text='BUTTON 2', size_hint=(None, None), pos=(50, 100), size=(200, 60),
                               color=(0, 1, 0, 1), font_size=dp(30), on_press=self.do_sum_2)

        self.fl = FloatLayout()

        self.bl = BoxLayout()
        self.bl.add_widget(tmp_btn)
        self.bl.add_widget(tmp_btn2)

        tabs = TabbedPanel()
        tabs.do_default_tab = False
        tab1 = HoverTabs(text='TAB 1')
        tab3 = HoverTabs(text='TAB 2', on_press=self.tab2_func)
        tab4 = HoverTabs(text='TAB 3')

        tab3.add_widget(self.bl)
        tabs.add_widget(tab1)
        tabs.add_widget(tab3)
        tabs.add_widget(tab4)
        tabs.default_tab = tab1
        self.fl.add_widget(tabs)

    def build(self):
        return self.fl

    def do_sum(self, *args, **kwargs):
        print("DOING SUM")

    def do_sum_2(self, *args, **kwargs):
        print("DOING SUM 2")

    def tab2_func(self, *args, **kwargs):
        print('CLICKED ON TAB2')

if __name__=='__main__':
    ButtonApp().run()