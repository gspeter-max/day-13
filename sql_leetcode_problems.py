''' 3436. Find Valid Emails ''' 
# Write your MySQL query statement below
select user_id, 
    email
from users
where email like  '%.com'
and email regexp '^[a-zA-Z0-9_]+@[a-zA-Z]+\\.com$'
order by user_id ; 

''' 1729. Find Followers Count ''' 
# Write your MySQL query statement below
select user_id, 
    count(follower_id) as followers_count
from followers
group by user_id
order by user_id ; 


''' 1731. The Number of Employees Which Report to Each Employee ''' 
with temp_temp as (
    select e1.employee_id, 
    e1.name,
    count(e2.reports_to) as reports_count, 
    round(avg(e2.age)) as average_age
    from employees e1 
    inner join employees e2  on e1.employee_id = e2.reports_to 
    group by e1.employee_id 
)
select * 
from temp_temp order by employee_id ; 
