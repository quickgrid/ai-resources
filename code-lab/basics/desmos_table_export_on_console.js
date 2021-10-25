/*
Exports a desmos table by running code below on browser developers tools console.

Reference,
https://www.reddit.com/r/desmos/comments/dfnpe2/anyway_to_exportcopypaste_data_out_of_a_desmos/
*/

state = Calc.getState()

for (let i = 0; i < state.expressions.list.length; i++) {
  if (state.expressions.list[i].type == "table") {
    for (let j = 0; j < state.expressions.list[i].columns.length; j++) {
      console.log(state.expressions.list[i].columns[j].latex + " = " + state.expressions.list[i].columns[j].values.toString())
    }
  }
}
