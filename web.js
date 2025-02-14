
const list = [
    {"name":"小明","age":25},
    {"name":"小红","age":24},
    {"name":"小刚","age":26}
];

const fetchUsers = (list)=>{
    for(const item of list){
        if(item.age >25){
            item.age = 25
        }
    }
}

console.log(list);
fetchUsers(list);
console.log(list);

// --------------
let age = 26;
const changeAge = (age)=>{
    if(age>25){
        age = 25;
    }
}
changeAge(age)
console.log(age)

// --------------

const doing = (back)=>{
    const data = {
        "name":"小明",
        "age":25
    }
    back(data);
}